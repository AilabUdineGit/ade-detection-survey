# Adapted from
# https://github.com/davidsbatista/NER-Evaluation

import logging
from collections import namedtuple
from copy import deepcopy
from tqdm import tqdm
import pandas as pd
import json
from spacy.lang.en import English
#from spacy.gold import biluo_tags_from_offsets as get_biluo
from spacy.training import offsets_to_biluo_tags as get_biluo
from sklearn.metrics import f1_score, precision_score, recall_score
from ast import literal_eval
from typing import Dict, NewType
from utils.interval_merger import IntervalMerger

Entity = namedtuple("Entity", "e_type start_offset end_offset")
Metrics = NewType("Metrics", Dict[str,Dict[str,float]])

class EvaluatorNER:
    def __init__(self, df, pred_col_name, gold_col_name):
        """
            Parameters
            ----------
            - df: pd.DataFrame
            - pred_col_name: str
                name of the dataframe's column containing the model's predictions (list(tuple))
            - gold_col_name: str
                name of the dataframe's column containing the gold labels (list(tuple))
        """
        nlp = English()
        #df[gold_col_name] = df[gold_col_name].apply(lambda x: literal_eval(x))
        #df[pred_col_name] = df[pred_col_name].apply(lambda x: literal_eval(x))
        df["spacy_doc"] = df.apply(lambda row: nlp(row.text),axis=1)
        
        df["gold_fix"] = df.apply(lambda x: IntervalMerger().merge([self.fix_ent(x.spacy_doc,y) for y in x[gold_col_name]]), axis=1)
        df["pred_fix"] = df.apply(lambda x: IntervalMerger().merge([self.fix_ent(x.spacy_doc,y) for y in x[pred_col_name]]), axis=1)
        
        df["gold_tags"] = df.apply(lambda x: get_biluo(x.spacy_doc, x.gold_fix), axis=1)
        df["pred_tags"] = df.apply(lambda x: get_biluo(x.spacy_doc, x.pred_fix), axis=1)
        
        df["gold_tags"] = df.apply(lambda x: [tag.replace("U-","B-").replace("L-","I-") for tag in x.gold_tags], axis=1)
        df["pred_tags"] = df.apply(lambda x: [tag.replace("U-","B-").replace("L-","I-") for tag in x.pred_tags], axis=1)
        
        all_ent_types = set([y.split("-")[-1] for x in df.pred_tags.tolist() for y in x]) - set("O")
        
        self.evaluator = Evaluator(df.gold_tags.tolist(), df.pred_tags.tolist(), all_ent_types)
        self.evaluator.evaluate()
    
    def get(self) -> Metrics:
        return self.evaluator.get()

    def merge_discontinuous_overlaps(self, sdocs, annotation_type):
        for sdoc in ptqdm(sdocs):
            for span in sdoc.doc.spans:
                if span.contains_annotation(annotation_type):
                    for i in span.intervals:
                        for j in span.intervals:
                            if i != j and i.overlaps(j):
                                i.begin = min(i.begin, j.begin)
                                i.end = max(i.end, j.end)
                                span.intervals.remove(j)
        return sdocs


    def fix_ent(self, doc, ent):
        
        #start,end,ent_type = ent
        start,end = ent
        
        for tok in doc:
            tstart = tok.idx
            tend = tok.idx + len(tok)
            
            if tstart < start < tend:
                start = tstart
                break
            elif start == tstart-1:
                start += 1
                break
                
        for tok in doc:
            tstart = tok.idx
            tend = tok.idx + len(tok)
            
            if tstart < end < tend:
                end = tend
                break
            elif end == tend+1:
                end -= 1
                break
                
        #return (start, end, ent_type)
        return (start, end, "ADR")


class Evaluator():

    def __init__(self, true, pred, tags="ADR"):
        """
            Parameters
            ----------
            - true: list(tuple): gold labels
            - pred: list(tuple): predicted labels
        """

        if len(true) != len(pred):
            raise ValueError("Number of predicted documents does not equal true")

        self.true = true
        self.pred = pred
        self.tags = tags

        # Setup dict into which metrics will be stored.
        fields = ['correct','incorrect','partial','missed','spurious','possible','actual','precision','recall','f1']
        self.metrics_results = {field:0 for field in fields}

        # Copy results dict to cover the four schemes.

        self.results = {
            'strict': deepcopy(self.metrics_results),
            'ent_type': deepcopy(self.metrics_results),
            'partial':deepcopy(self.metrics_results),
            'exact':deepcopy(self.metrics_results),
            }

        # Create an accumulator to store results

        self.evaluation_agg_entities_type = {e: deepcopy(self.results) for e in tags}


    def evaluate(self):

        logging.info(
            "Imported %s predictions for %s true examples",
            len(self.pred), len(self.true)
        )

        for true_ents, pred_ents in tqdm(zip(self.true, self.pred), total=len(self.true)):

            # Check that the length of the true and predicted examples are the
            # same. This must be checked here, because another error may not
            # be thrown if the lengths do not match.

            if len(true_ents) != len(pred_ents):
                raise ValueError("Prediction length does not match true example length")

            # Compute results for one message

            tmp_results, tmp_agg_results = compute_metrics(
                collect_named_entities(true_ents),
                collect_named_entities(pred_ents),
                self.tags
            )

            # Cycle through each result and accumulate

            # TODO: Combine these loops below:

            for eval_schema in self.results:

                for metric in self.results[eval_schema]:

                    self.results[eval_schema][metric] += tmp_results[eval_schema][metric]

            # Calculate global precision and recall

            self.results = compute_precision_recall_wrapper(self.results)

            # Aggregate results by entity type

            for e_type in self.tags:

                for eval_schema in tmp_agg_results[e_type]:

                    for metric in tmp_agg_results[e_type][eval_schema]:

                        self.evaluation_agg_entities_type[e_type][eval_schema][metric] += tmp_agg_results[e_type][eval_schema][metric]

                # Calculate precision recall at the individual entity level

                self.evaluation_agg_entities_type[e_type] = compute_precision_recall_wrapper(self.evaluation_agg_entities_type[e_type])

        #self.print_metrics(self.results, "partial")
        #self.print_metrics(self.results, "strict")
        #return self.results, self.evaluation_agg_entities_type

    def get(self):
        return self.results

    def print_metrics(self, all_metrics, metrics_type):
        metrics = ["precision", "recall", "f1"]
        res = {m:-1 for m in metrics}
        
        print("-----------------------")
        print(f"NER {metrics_type} Metrics")
        for m in metrics:
            res[m] = all_metrics[metrics_type][m]
            print(f"{m:<10}| {round(all_metrics[metrics_type][m],4)}")

def collect_named_entities(tokens):
    """
    Creates a list of Entity named-tuples, storing the entity type and the start and end
    offsets of the entity.

    :param tokens: a list of tags
    :return: a list of Entity named-tuples
    """

    named_entities = []
    start_offset = None
    end_offset = None
    ent_type = None

    for offset, token_tag in enumerate(tokens):

        if token_tag == 'O':
            if ent_type is not None and start_offset is not None:
                end_offset = offset - 1
                named_entities.append(Entity(ent_type, start_offset, end_offset))
                start_offset = None
                end_offset = None
                ent_type = None

        elif ent_type is None:
            ent_type = token_tag[2:]
            start_offset = offset

        elif ent_type != token_tag[2:] or (ent_type == token_tag[2:] and token_tag[:1] == 'B'):

            end_offset = offset - 1
            named_entities.append(Entity(ent_type, start_offset, end_offset))

            # start of a new entity
            ent_type = token_tag[2:]
            start_offset = offset
            end_offset = None

    # catches an entity that goes up until the last token
    
    if (ent_type is not None) and (start_offset is not None) and end_offset is None:
        named_entities.append(Entity(ent_type, start_offset, len(tokens)-1))

    return named_entities


def compute_metrics(true_named_entities, pred_named_entities, tags):


    eval_metrics = {'correct': 0, 'incorrect': 0, 'partial': 0, 'missed': 0, 'spurious': 0, 'precision': 0, 'recall': 0, 'f1': 0}

    # overall results
    
    evaluation = {
        'strict': deepcopy(eval_metrics),
        'ent_type': deepcopy(eval_metrics),
        'partial': deepcopy(eval_metrics),
        'exact': deepcopy(eval_metrics)
    }

    # results by entity type

    evaluation_agg_entities_type = {e: deepcopy(evaluation) for e in tags}

    # keep track of entities that overlapped

    true_which_overlapped_with_pred = []

    # Subset into only the tags that we are interested in.
    # NOTE: we remove the tags we don't want from both the predicted and the
    # true entities. This covers the two cases where mismatches can occur:
    #
    # 1) Where the model predicts a tag that is not present in the true data
    # 2) Where there is a tag in the true data that the model is not capable of
    # predicting.

    true_named_entities = [ent for ent in true_named_entities if ent.e_type in tags]
    pred_named_entities = [ent for ent in pred_named_entities if ent.e_type in tags]

    # go through each predicted named-entity

    for pred in pred_named_entities:
        found_overlap = False

        # Check each of the potential scenarios in turn. See
        # http://www.davidsbatista.net/blog/2018/05/09/Named_Entity_Evaluation/
        # for scenario explanation.

        # Scenario I: Exact match between true and pred

        if pred in true_named_entities:
            true_which_overlapped_with_pred.append(pred)
            evaluation['strict']['correct'] += 1
            evaluation['ent_type']['correct'] += 1
            evaluation['exact']['correct'] += 1
            evaluation['partial']['correct'] += 1

            # for the agg. by e_type results
            evaluation_agg_entities_type[pred.e_type]['strict']['correct'] += 1
            evaluation_agg_entities_type[pred.e_type]['ent_type']['correct'] += 1
            evaluation_agg_entities_type[pred.e_type]['exact']['correct'] += 1
            evaluation_agg_entities_type[pred.e_type]['partial']['correct'] += 1

        else:

            # check for overlaps with any of the true entities

            for true in true_named_entities:

                pred_range = range(pred.start_offset, pred.end_offset)
                true_range = range(true.start_offset, true.end_offset)
                
                pred_range = range(pred.start_offset, pred.end_offset+1)
                true_range = range(true.start_offset, true.end_offset+1)

                # Scenario IV: Offsets match, but entity type is wrong

                if true.start_offset == pred.start_offset and pred.end_offset == true.end_offset \
                        and true.e_type != pred.e_type:

                    # overall results
                    evaluation['strict']['incorrect'] += 1
                    evaluation['ent_type']['incorrect'] += 1
                    evaluation['partial']['correct'] += 1
                    evaluation['exact']['correct'] += 1

                    # aggregated by entity type results
                    evaluation_agg_entities_type[true.e_type]['strict']['incorrect'] += 1
                    evaluation_agg_entities_type[true.e_type]['ent_type']['incorrect'] += 1
                    evaluation_agg_entities_type[true.e_type]['partial']['correct'] += 1
                    evaluation_agg_entities_type[true.e_type]['exact']['correct'] += 1

                    true_which_overlapped_with_pred.append(true)
                    found_overlap = True

                    break

                # check for an overlap i.e. not exact boundary match, with true entities

                elif find_overlap(true_range, pred_range):

                    true_which_overlapped_with_pred.append(true)

                    # Scenario V: There is an overlap (but offsets do not match
                    # exactly), and the entity type is the same.
                    # 2.1 overlaps with the same entity type

                    if pred.e_type == true.e_type:

                        # overall results
                        evaluation['strict']['incorrect'] += 1
                        evaluation['ent_type']['correct'] += 1
                        evaluation['partial']['partial'] += 1
                        evaluation['exact']['incorrect'] += 1

                        # aggregated by entity type results
                        evaluation_agg_entities_type[true.e_type]['strict']['incorrect'] += 1
                        evaluation_agg_entities_type[true.e_type]['ent_type']['correct'] += 1
                        evaluation_agg_entities_type[true.e_type]['partial']['partial'] += 1
                        evaluation_agg_entities_type[true.e_type]['exact']['incorrect'] += 1

                        found_overlap = True

                        break

                    # Scenario VI: Entities overlap, but the entity type is
                    # different.

                    else:
                        # overall results
                        evaluation['strict']['incorrect'] += 1
                        evaluation['ent_type']['incorrect'] += 1
                        evaluation['partial']['partial'] += 1
                        evaluation['exact']['incorrect'] += 1

                        # aggregated by entity type results
                        # Results against the true entity

                        evaluation_agg_entities_type[true.e_type]['strict']['incorrect'] += 1
                        evaluation_agg_entities_type[true.e_type]['partial']['partial'] += 1
                        evaluation_agg_entities_type[true.e_type]['ent_type']['incorrect'] += 1
                        evaluation_agg_entities_type[true.e_type]['exact']['incorrect'] += 1

                        # Results against the predicted entity

                        # evaluation_agg_entities_type[pred.e_type]['strict']['spurious'] += 1

                        found_overlap = True

                        break

            # Scenario II: Entities are spurious (i.e., over-generated).

            if not found_overlap:

                # Overall results

                evaluation['strict']['spurious'] += 1
                evaluation['ent_type']['spurious'] += 1
                evaluation['partial']['spurious'] += 1
                evaluation['exact']['spurious'] += 1

                # Aggregated by entity type results

                # NOTE: when pred.e_type is not found in tags
                # or when it simply does not appear in the test set, then it is
                # spurious, but it is not clear where to assign it at the tag
                # level. In this case, it is applied to all target_tags
                # found in this example. This will mean that the sum of the
                # evaluation_agg_entities will not equal evaluation.

                for true in tags:                    

                    evaluation_agg_entities_type[true]['strict']['spurious'] += 1
                    evaluation_agg_entities_type[true]['ent_type']['spurious'] += 1
                    evaluation_agg_entities_type[true]['partial']['spurious'] += 1
                    evaluation_agg_entities_type[true]['exact']['spurious'] += 1

    # Scenario III: Entity was missed entirely.

    for true in true_named_entities:
        if true in true_which_overlapped_with_pred:
            continue
        else:
            # overall results
            evaluation['strict']['missed'] += 1
            evaluation['ent_type']['missed'] += 1
            evaluation['partial']['missed'] += 1
            evaluation['exact']['missed'] += 1

            # for the agg. by e_type
            evaluation_agg_entities_type[true.e_type]['strict']['missed'] += 1
            evaluation_agg_entities_type[true.e_type]['ent_type']['missed'] += 1
            evaluation_agg_entities_type[true.e_type]['partial']['missed'] += 1
            evaluation_agg_entities_type[true.e_type]['exact']['missed'] += 1

    # Compute 'possible', 'actual' according to SemEval-2013 Task 9.1 on the
    # overall results, and use these to calculate precision and recall.

    for eval_type in evaluation:
        evaluation[eval_type] = compute_actual_possible(evaluation[eval_type])

    # Compute 'possible', 'actual', and precision and recall on entity level
    # results. Start by cycling through the accumulated results.

    for entity_type, entity_level in evaluation_agg_entities_type.items():

        # Cycle through the evaluation types for each dict containing entity
        # level results.

        for eval_type in entity_level:

            evaluation_agg_entities_type[entity_type][eval_type] = compute_actual_possible(
                entity_level[eval_type]
            )

    return evaluation, evaluation_agg_entities_type


def find_overlap(true_range, pred_range):
    """Find the overlap between two ranges

    Find the overlap between two ranges. Return the overlapping values if
    present, else return an empty set().

    Examples:

    >>> find_overlap((1, 2), (2, 3))
    2
    >>> find_overlap((1, 2), (3, 4))
    set()
    """

    true_set = set(true_range)
    pred_set = set(pred_range)

    overlaps = true_set.intersection(pred_set)

    return overlaps


def compute_actual_possible(results):
    """
    Takes a result dict that has been output by compute metrics.
    Returns the results dict with actual, possible populated.

    When the results dicts is from partial or ent_type metrics, then
    partial_or_type=True to ensure the right calculation is used for
    calculating precision and recall.
    """

    correct = results['correct']
    incorrect = results['incorrect']
    partial = results['partial']
    missed = results['missed']
    spurious = results['spurious']

    # Possible: number annotations in the gold-standard which contribute to the
    # final score

    possible = correct + incorrect + partial + missed

    # Actual: number of annotations produced by the NER system

    actual = correct + incorrect + partial + spurious

    results["actual"] = actual
    results["possible"] = possible

    return results


def compute_precision_recall(results, partial_or_type=False):
    """
    Takes a result dict that has been output by compute metrics.
    Returns the results dict with precison and recall populated.

    When the results dicts is from partial or ent_type metrics, then
    partial_or_type=True to ensure the right calculation is used for
    calculating precision and recall.
    """

    actual = results["actual"]
    possible = results["possible"]
    partial = results['partial']
    correct = results['correct']

    if partial_or_type:
        precision = (correct + 0.5 * partial) / actual if actual > 0 else 0
        recall = (correct + 0.5 * partial) / possible if possible > 0 else 0

    else:
        precision = correct / actual if actual > 0 else 0
        recall = correct / possible if possible > 0 else 0

    results["precision"] = precision
    results["recall"] = recall
    results["f1"] = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    return results


def compute_precision_recall_wrapper(results):
    """
    Wraps the compute_precision_recall function and runs on a dict of results
    """

    results_a = {key: compute_precision_recall(value, True) for key, value in results.items() if
                 key in ['partial', 'ent_type']}
    results_b = {key: compute_precision_recall(value) for key, value in results.items() if
                 key in ['strict', 'exact']}

    results = {**results_a, **results_b}

    return results




