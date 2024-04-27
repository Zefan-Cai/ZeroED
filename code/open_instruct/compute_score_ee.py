import json
import datasets

import numpy as np

from sklearn.metrics import (
    classification_report
)


# @datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class EE(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            description="hello",
            citation="hello",
            inputs_description="hello",
            features=datasets.Features(
                {
                    "event_type": datasets.Value("int32"),
                    "data_id": datasets.Value("int32"),
                    "predictions": datasets.Value("int32"),
                    "references": datasets.Value("int32")
                }
            ),
            codebase_urls=["hello"],
            reference_urls=["hello"]
        )

    def _compute(self, predictions, references, event_type, data_id):
        
        scores = self._ee_scores_debug(predictions, references, event_type, data_id)
        
        return scores
    
    def _accuracy(self, predictions, references):
        Correct = 0
        for pred, gt in zip(predictions, references):
            if pred == gt:
                Correct += 1 
        return Correct/len(predictions)
    
    
    def _ee_scores_debug(self, predictions, references, event_type, data_id):
        
        group_predictions = []
        group_references = []
        # Basically sort the results by data_id here
        distinct_data_id = list(set(data_id))
        all_predictions = []
        all_references = []
        all_event_type = []
        for id in range(max(distinct_data_id)+1):
            all_predictions.append([])
            all_references.append([])
            all_event_type.append([])
        
        for pred, gt, event, id in zip(predictions, references, event_type, data_id):
            if event not in all_event_type[id]:
                all_predictions[id].append(pred)
                all_references[id].append(gt)
                all_event_type[id].append(event)

        # get max length. Don't know what this line for...
        len_events = max([len(item) for item in all_predictions])
        # for item in all_predictions:
        #     if len(item) > len_events:
        #         len_events = len(item)

        count = 0
        error_count = 0
        for index_i in range(len(all_predictions)):
            if len(all_predictions[index_i]) < len_events:
                error_count += 1
                continue
            count += 1
            
            temp_predictions = []
            temp_references = []
            
            # print(f"debug all_predictions {index_i} {all_predictions[index_i]}")
            # print(f"debug all_references {index_i} {all_references[index_i]}")
            # print(f"debug all_event_type {index_i} {all_event_type[index_i]}")
            
            for index_j in range(0, len_events):
                
                # What's the meaning of this code
                
                # print(f"debug len(predictions) {len(predictions)} index_i {index_i} index_j {index_j}")
                
                if all_predictions[index_i][index_j] != 32000:
                    temp_predictions.append((all_predictions[index_i][index_j], all_event_type[index_i][index_j]))
                    
            # for index_j in range(0, len_events):
                if all_references[index_i][index_j] != 32000:
                    temp_references.append((all_references[index_i][index_j], all_event_type[index_i][index_j]))
                    
            group_predictions.append(tuple(temp_predictions))
            group_references.append(tuple(temp_references))

        print(f"debug all_predictions with len_events events {count} error_count {error_count}")

        # print(f"debug len(group_predictions) {group_predictions[:20]}")
        # print(f"debug len(group_references) {group_references[:20]}")

        # tri_id
        gold_tri_id_num, pred_tri_id_num, match_tri_id_num = 0, 0, 0
        
        for gold_trigger, pred_trigger in zip(group_references, group_predictions):
            if gold_trigger != [] and pred_trigger != []:
                # print([t[0][0] for t in gold_trigger])
                gold_set = set(tuple([t[0] for t in gold_trigger]))
                pred_set = set(tuple([t[0] for t in pred_trigger]))
                # print(gold_trigger)
                # print(pred_trigger)
                # print(gold_set)
                # print(pred_set)
                # print(gold_set & pred_set)
            else:
                gold_set = set(gold_trigger)
                pred_set = set(pred_trigger)
            # gold_set = set(gold_trigger)
            # pred_set = set(pred_trigger)
            # print(pred_set)
            gold_tri_id_num += len(gold_set)
            pred_tri_id_num += len(pred_set)
            match_tri_id_num += len(gold_set & pred_set)
        
        # tri_cls
        gold_tri_cls_num, pred_tri_cls_num, match_tri_cls_num = 0, 0, 0
        for gold_trigger, pred_trigger in zip(group_references, group_predictions):
            
            # print("debug")
            # print(pred_trigger)
            # print(gold_trigger)
            
            gold_set = set(gold_trigger)
            pred_set = set(pred_trigger)
            gold_tri_cls_num += len(gold_set)
            pred_tri_cls_num += len(pred_set)
            match_tri_cls_num += len(gold_set & pred_set)
        
        scores = {
            'tri_id': (gold_tri_id_num, pred_tri_id_num, match_tri_id_num) + self.compute_f1(pred_tri_id_num, gold_tri_id_num, match_tri_id_num),
            'tri_cls': (gold_tri_cls_num, pred_tri_cls_num, match_tri_cls_num) + self.compute_f1(pred_tri_cls_num, gold_tri_cls_num, match_tri_cls_num),
        }
        
        return {
            "gold_tri_id_num": scores['tri_id'][0],
            "pred_tri_id_num": scores['tri_id'][1],
            "match_tri_id_num": scores['tri_id'][2],
            "trigger_id_precision": scores['tri_id'][3],
            "trigger_id_recall": scores['tri_id'][4],
            "trigger_id_f1": scores['tri_id'][5],
            "gold_tri_cls_num": scores['tri_cls'][0],
            "pred_tri_cls_num": scores['tri_cls'][1],
            "match_tri_cls_num": scores['tri_cls'][2],
            "trigger_cls_precision": scores['tri_cls'][3],
            "trigger_cls_recall": scores['tri_cls'][4],
            "trigger_cls_f1": scores['tri_cls'][5],
            }

    
    
    
    
    
    
    
    
    
    
    def _ee_scores(self, predictions, references, event_type):
        
        group_predictions = []
        group_references = []
        
        first_0_idx = event_type.index(0)
        
        # print(f"debug len(predictions) {len(predictions)}")
        
        # print(f"debug predictions {predictions[:100]}")
        # print(f"debug references {references[:100]}")
        # print(f"debug event_type {event_type[:100]}")
        # print(f"debug first_0_idx {first_0_idx}")
        
        num_samples = (len(predictions) - first_0_idx) // 33
        
        for index_i in range(0, num_samples):
            
            temp_predictions = []
            temp_references = []
            
            store_event_id_debug = []
            
            # print(f"debug first_0_idx {first_0_idx} {event_type[first_0_idx]} index_i {index_i} first_0_idx + index_i * 33 {first_0_idx + index_i * 33}")
            # print(f"debug event_type[first_0_idx + index_i * 33] {event_type[first_0_idx + index_i * 33: first_0_idx + index_i * 33 + 33]}")
            
            for index_j in range(0, 33):
                
                # What's the meaning of this code
                
                # print(f"debug len(predictions) {len(predictions)} index_i {index_i} index_j {index_j}")
                
                if predictions[first_0_idx + index_i * 33 + index_j] != 32000:
                    temp_predictions.append((predictions[first_0_idx + index_i * 33 + index_j], event_type[first_0_idx + index_i * 33 + index_j]))
                    
            # for index_j in range(0, 33):
                if references[first_0_idx + index_i * 33 + index_j] != 32000:
                    temp_references.append((references[first_0_idx + index_i * 33 + index_j], event_type[first_0_idx + index_i * 33 + index_j]))
                    
            group_predictions.append(tuple(temp_predictions))
            group_references.append(tuple(temp_references))

        # print(f"debug len(group_predictions) {group_predictions[:20]}")
        # print(f"debug len(group_references) {group_references[:20]}")

        # tri_id
        gold_tri_id_num, pred_tri_id_num, match_tri_id_num = 0, 0, 0
        
        for gold_trigger, pred_trigger in zip(group_references, group_predictions):
            if gold_trigger != [] and pred_trigger != []:
                # print([t[0][0] for t in gold_trigger])
                gold_set = set(tuple([t[0] for t in gold_trigger]))
                pred_set = set(tuple([t[0] for t in pred_trigger]))
                # print(gold_trigger)
                # print(pred_trigger)
                # print(gold_set)
                # print(pred_set)
                # print(gold_set & pred_set)
            else:
                gold_set = set(gold_trigger)
                pred_set = set(pred_trigger)
            # gold_set = set(gold_trigger)
            # pred_set = set(pred_trigger)
            gold_tri_id_num += len(gold_set)
            pred_tri_id_num += len(pred_set)
            match_tri_id_num += len(gold_set & pred_set)
        
        # tri_cls
        gold_tri_cls_num, pred_tri_cls_num, match_tri_cls_num = 0, 0, 0
        for gold_trigger, pred_trigger in zip(group_references, group_predictions):
            
            # print("debug")
            # print(pred_trigger)
            # print(gold_trigger)
            
            gold_set = set(gold_trigger)
            pred_set = set(pred_trigger)
            gold_tri_cls_num += len(gold_set)
            pred_tri_cls_num += len(pred_set)
            match_tri_cls_num += len(gold_set & pred_set)
        
        scores = {
            'tri_id': (gold_tri_id_num, pred_tri_id_num, match_tri_id_num) + self.compute_f1(pred_tri_id_num, gold_tri_id_num, match_tri_id_num),
            'tri_cls': (gold_tri_cls_num, pred_tri_cls_num, match_tri_cls_num) + self.compute_f1(pred_tri_cls_num, gold_tri_cls_num, match_tri_cls_num),
        }
        
        return {
            "gold_tri_id_num": scores['tri_id'][0],
            "pred_tri_id_num": scores['tri_id'][1],
            "match_tri_id_num": scores['tri_id'][2],
            "trriger_id_precision": scores['tri_id'][3],
            "trriger_id_recall": scores['tri_id'][4],
            "trriger_id_f1": scores['tri_id'][5],
            "gold_tri_cls_num": scores['tri_cls'][0],
            "pred_tri_cls_num": scores['tri_cls'][1],
            "match_tri_cls_num": scores['tri_cls'][2],
            "trriger_cls_precision": scores['tri_cls'][3],
            "trriger_cls_recall": scores['tri_cls'][4],
            "trriger_cls_f1": scores['tri_cls'][5],
            }

    
    def _scores(self, predictions, references):
        

            
        
        total_labels = []
        total_prelabels = []
        
        Correct = 0
        
        for pred, gt in zip(predictions, references):
            if gt == 32000:
                total_labels.append(0)
                if pred == 32000:
                    total_prelabels.append(0)
                    print(f"debug 0 0 pred: {pred} gt: {gt}")
                else:
                    total_prelabels.append(1)
                    print(f"debug 0 1 pred: {pred} gt: {gt}")
            else:
                total_labels.append(1)
                if pred == gt:
                    total_prelabels.append(1)
                    print(f"debug 1 1 pred: {pred} gt: {gt}")
                else:
                    total_prelabels.append(0)
                    print(f"debug 1 -1 pred: {pred} gt: {gt}")
            if pred == gt:
                Correct += 1
    
        
        total_prelabels = np.array(total_prelabels)
        total_labels = np.array(total_labels)
                
        classifi_report = classification_report(
            total_labels, total_prelabels, target_names=[0, 1], output_dict=True
        )
        positive_precision = classifi_report[1]["precision"]
        positive_recall = classifi_report[1]["recall"]
        positive_f1_score = classifi_report[1]["f1-score"]
        negative_precision = classifi_report[0]["precision"]
        negative_recall = classifi_report[0]["recall"]
        negative_f1 = classifi_report[0]["f1-score"]

        gold_tri_id_num, pred_tri_id_num, match_tri_id_num = 0, 0, 0
        
        # tri_id
        for gold_trigger, pred_trigger in zip(references, predictions):

            if gold_trigger == pred_trigger:
                match_tri_id_num += 1

            gold_tri_id_num += 1
            pred_tri_id_num += 1
        
        # tri_cls
        # gold_tri_cls_num, pred_tri_cls_num, match_tri_cls_num = 0, 0, 0
        # for gold_trigger, pred_trigger in zip(references, predictions):
        #     gold_set = set(gold_trigger)
        #     pred_set = set(pred_trigger)
        #     gold_tri_cls_num += len(gold_set)
        #     pred_tri_cls_num += len(pred_set)
        #     match_tri_cls_num += len(gold_set & pred_set)
            
        scores = {
            'tri_id': (gold_tri_id_num, pred_tri_id_num, match_tri_id_num) + self.compute_f1(pred_tri_id_num, gold_tri_id_num, match_tri_id_num),
            # 'tri_cls': (gold_tri_cls_num, pred_tri_cls_num, match_tri_cls_num) + self.compute_f1(pred_tri_cls_num, gold_tri_cls_num, match_tri_cls_num),
        }
                
        return {
            "accuracy": Correct / len(predictions),
            "Positive_Precision": positive_precision,
            "Positive_Recall": positive_recall,
            "Positive_F1": positive_f1_score,
            "Negative_Precision": negative_precision,
            "Negative_Recall": negative_recall,
            "Negative_F1": negative_f1,
            "gold_tri_id_num": scores['tri_id'][0],
            "pred_tri_id_num": scores['tri_id'][1],
            "match_tri_id_num": scores['tri_id'][2],
            "trriger_id_precision": scores['tri_id'][3],
            "trriger_id_recall": scores['tri_id'][4],
            "trriger_id_f1": scores['tri_id'][5],
            # "gold_tri_cls_num": scores['tri_cls'][0],
            # "pred_tri_cls_num": scores['tri_cls'][1],
            # "match_tri_cls_num": scores['tri_cls'][2],
            # "trriger_cls_precision": scores['tri_cls'][3],
            # "trriger_cls_recall": scores['tri_cls'][4],
            # "trriger_cls_f1": scores['tri_cls'][5],
            }
                
    def compute_f1(self, predicted, gold, matched):
        precision = self.safe_div(matched, predicted)
        recall = self.safe_div(matched, gold)
        f1 = self.safe_div(2 * precision * recall, precision + recall)
        return precision, recall, f1


    def safe_div(self, num, denom):
        if denom > 0:
            return num / denom
        else:
            return 0
        
    