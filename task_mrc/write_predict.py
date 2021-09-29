# coding=utf-8
from __future__ import absolute_import, division, print_function
import json
import logging
import math
import collections
from io import open
from transformers.models.bert.tokenization_bert import BasicTokenizer


logger = logging.getLogger(__name__)


def write_predictions(all_examples, all_features, all_results, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file, args):
    """ 如果需要，将最终预测写入json文件并记录为空的概率。 """
    logger.info("Writing predictions to: %s" % output_prediction_file)
    logger.info("Writing nbest to: %s" % output_nbest_file)

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit"]
    )

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]
        prelim_predictions = []

        # 跟踪零位起始和结束位置的最小得分
        score_null = 1000000  # large and positive
        min_null_feature_index = 0  # the paragraph slice with min null score
        null_start_logit = 0  # the start logit at the slice with min null score
        null_end_logit = 0  # the end logit at the slice with min null score
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            start_indexes = _get_best_indexes(result.start_logits, args.n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, args.n_best_size)
            # if we could have irrelevant answers, get the min score of irrelevant
            if args.version_2_with_negative:
                feature_null_score = result.start_logits[0] + result.end_logits[0]
                if feature_null_score < score_null:
                    score_null = feature_null_score
                    min_null_feature_index = feature_index
                    null_start_logit = result.start_logits[0]
                    null_end_logit = result.end_logits[0]
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(
                            start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > args.max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(feature_index=feature_index, start_index=start_index,
                                          end_index=end_index, start_logit=result.start_logits[start_index],
                                          end_logit=result.end_logits[end_index])
                    )
        if args.version_2_with_negative:
            prelim_predictions.append(
                _PrelimPrediction(feature_index=min_null_feature_index, start_index=0, end_index=0,
                                  start_logit=null_start_logit, end_logit=null_end_logit))
        prelim_predictions = sorted(prelim_predictions,
                                    key=lambda x: (x.start_logit + x.end_logit),
                                    reverse=True)

        _NbestPrediction = collections.namedtuple("NbestPrediction", ["text", "start_logit", "end_logit"])
        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= args.n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index > 0:  # this is a non-null prediction
                tok_tokens = feature.tokens[pred.start_index:(
                    pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.context[orig_doc_start:(
                    orig_doc_end + 1)]
                tok_text = " ".join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text = get_final_text(tok_text, orig_text, args.do_lower_case, args.verbose_logging)
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True

            nbest.append(_NbestPrediction(text=final_text, start_logit=pred.start_logit,
                                          end_logit=pred.end_logit))
        # if we didn't include the empty option in the n-best, include it
        if args.version_2_with_negative:
            if "" not in seen_predictions:
                nbest.append(_NbestPrediction(text="", start_logit=null_start_logit,
                                              end_logit=null_end_logit))

            # In very rare edge cases we could only have single null prediction.
            # So we just create a nonce prediction in this case to avoid failure.
            if len(nbest) == 1:
                nbest.insert(0, _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(_NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        assert len(nbest) >= 1

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        if not args.version_2_with_negative:
            all_predictions[example.qas_id] = nbest_json[0]["text"]
        else:
            # predict "" iff the null score - the score of best non-null > threshold
            score_diff = score_null - best_non_null_entry.start_logit - best_non_null_entry.end_logit
            scores_diff_json[example.qas_id] = score_diff
            if score_diff > args.null_score_diff_threshold:
                all_predictions[example.qas_id] = ""
            else:
                all_predictions[example.qas_id] = best_non_null_entry.text
        all_nbest_json[example.qas_id] = nbest_json

    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4) + "\n")
    with open(output_nbest_file, "w") as writer:
        writer.write(json.dumps(all_nbest_json, indent=4) + "\n")
    if args.version_2_with_negative:
        with open(output_null_log_odds_file, "w") as writer:
            writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

    return all_predictions



def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    """ 将标记化的预测投影回原始文本 """
    # 创建数据时，跟踪原始数据之间的对齐(空白标记化)标记和WordPiece标记化标记。
    # 所以，'orig_text'包含了对应于预测的跨度。
    # 然而，'orig_text'可能包含我们不想在预测中使用的额外字符。
    # 例子:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    # 不返回'pred_text'，因为它已经被规范化了
    # 不返回'orig_text'，因为它包含了额外的 "'s".
    # 因此，必须应用半复杂的对齐启发式'pred_text'和'orig_text'以获得字符对字符的对齐。
    # 这在某些情况下可能会失败，在这种情况下我们只需要返回'orig_text'

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # 首先标记'orig_text'，从结果中去掉空白和'pred_text'，并检查它们的长度是否相同。
    # 如果长度不一样，则启发式失败。如果长度相同，则假设字符是一对一对齐的。
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
    tok_text = " ".join(tokenizer.tokenize(orig_text))
    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            logger.info(
                "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)
    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logger.info(
                "Length not equal after stripping spaces: '%s' vs '%s'",
                orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose_logging:
            logger.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            logger.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(
        enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs