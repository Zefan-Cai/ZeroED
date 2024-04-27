import argparse
import os
import re
import json
import tqdm
import glob
import torch
import random
import evaluate
from open_instruct.eval.utils import load_hf_lm_and_tokenizer, generate_completions
# query_openai_chat_model

def safe_div(num, denom):
    if denom > 0:
        return num / denom
    else:
        return 0

def compute_f1(predicted, gold, matched):
    precision = safe_div(matched, predicted)
    recall = safe_div(matched, gold)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return precision, recall, f1


# def cal_scores(gold_triggers, pred_triggers, gold_events, pred_events):
def cal_scores(gold_triggers, pred_triggers):
    assert len(gold_triggers) == len(pred_triggers)
    # assert len(gold_events) == len(pred_events)  
    
    
    # tri_id
    gold_tri_id_num, pred_tri_id_num, match_tri_id_num = 0, 0, 0
    
    # print(f"debug: {gold_triggers[2]}")
    # print(f"debug: {pred_triggers[2]}")
    
    # for gold_trigger, pred_trigger in zip(gold_triggers, pred_triggers):
    for gold_trigger, pred_trigger in zip(gold_triggers, pred_triggers):
        # fix bug
        gold_set = set(tuple([t[0] for t in gold_trigger])) if gold_trigger != [] else set([])
        pred_set = set(tuple([t[0] for t in pred_trigger])) if pred_trigger != [] else set([])
        gold_tri_id_num += len(gold_set)
        pred_tri_id_num += len(pred_set)
        match_tri_id_num += len(gold_set & pred_set)
    
    # tri_cls
    gold_tri_cls_num, pred_tri_cls_num, match_tri_cls_num = 0, 0, 0
    for gold_trigger, pred_trigger in zip(gold_triggers, pred_triggers):
        
        # print("debug")
        # print(pred_trigger)
        # print(gold_trigger)
        
        gold_set = set(gold_trigger)
        pred_set = set(pred_trigger)
        gold_tri_cls_num += len(gold_set)
        pred_tri_cls_num += len(pred_set)
        match_tri_cls_num += len(gold_set & pred_set)
    
    scores = {
        'tri_id': (gold_tri_id_num, pred_tri_id_num, match_tri_id_num) + compute_f1(pred_tri_id_num, gold_tri_id_num, match_tri_id_num),
        'tri_cls': (gold_tri_cls_num, pred_tri_cls_num, match_tri_cls_num) + compute_f1(pred_tri_cls_num, gold_tri_cls_num, match_tri_cls_num),
    }
    
    return scores

def get_trigger(examples):
    # test_gold_triggers, test_gold_events, test_pred_triggers, test_pred_events = [], [], [], []

    test_gold_object, test_pred_object = [], []


    for example in examples:
        # pred_object = []
        gt_trigger_object = []
        gt_event_object = []
        
        my_gold_object = []
        my_pred_object = []
        
        for sub_example in example:
            # print(sub_example["Event type"])
            
            event_type = sub_example["Event type"]
            trigger = sub_example["trigger"]
            
            raw_output = sub_example["prediction"]
            
            if trigger != '<trigger>':
                for tri in trigger:
                    gt_trigger_object.extend(tri)
                    gt_event_object.append(event_type)
                    my_gold_object.append((tri, event_type))

            
            try:
                # triggers = raw_output.split("Event trigger is ")[1].replace(".", "").strip().split(",")
                # Since we added the "Event trigger is" in the prompt during training, we change the parsing a bit
                triggers = raw_output.replace("Event trigger is ", "").replace(".", "").strip().split(",")
                # triggers = triggers.split('.')[0]
                # triggers = triggers.split(' and ')
                
                
                for t_cnt, t in enumerate(triggers):
                    if t != '<trigger>' and t != "":
                        # pred_object.append((t, event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
                        # pred_object.append(t, event_type) # (text, type, kwargs)
                        my_pred_object.append((t, event_type))
                        # print(t, event_type)
                    
                        
            except Exception as e:
                print(e)
                pass
            
            # if len(trigger) > 1 and trigger != '<trigger>':
            #     print(f"subexample trigger {trigger}")
            #     print(f"subexample event_type {event_type}")
            #     print(f"subexample triggers {triggers}")
            
            # Warning: What's this line doing?
            # sub_example["trigger"] = pred_object
                
        # pred_trigger_object = []
        # pred_event_object = []
        
        # for obj in pred_object:
        #     pred_event_object.append(obj[1])
        #     pred_trigger_object.append(obj[0])

        test_gold_object.append(tuple(my_gold_object))
        test_pred_object.append(tuple(my_pred_object))
        
        # print(f"debug my_gold_object {my_gold_object}")
        # print(f"debug my_pred_object {my_pred_object}")



        # test_gold_triggers.append(gt_trigger_object)
        # test_gold_events.append(gt_event_object)
        # test_pred_triggers.append(pred_trigger_object)
        # test_pred_events.append(pred_event_object)

    # print("debug")
    # print(test_pred_object[:10])
    # print(test_gold_object[:10])

    return test_gold_object, test_pred_object
    # return test_gold_object, test_pred_object, test_gold_events, test_pred_events
 

exact_match = evaluate.load("exact_match")

@torch.no_grad()
def eval_hf_model(args, model, tokenizer, examples, task_prompt, save_path=None):
    # targets = [example["target"] for example in examples]
    if save_path:
        fout = open(save_path, "w")

    prompts = []
    for example in examples:
        for sub_example in example:
            if args.use_chat_format:
                prompt = "<|user|>\n" + task_prompt.strip() + "\n\nQ: " + example["input"] + "\n<|assistant|>\nA:"
            else:
                prompt = sub_example["prompt"].strip()
                # prompt = task_prompt.strip() + "\n\nQ: " + example["prompt"] + "\nA:"
            prompts.append(prompt)

    # if args.no_cot:
    #     stop_sequnce = tokenizer.encode("\n\n", add_special_tokens=False)[-2:] # get the last token because the tokenizer may add space tokens at the start.
    # else:
    #     # let's not use the stop sequence for cot now since it's too inefficient when the generation is long. 
    #     # instead, we'll do some post-processing to extract the answer.
    #     stop_sequnce = None
    
    outputs = generate_completions(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        max_new_tokens=12,
        batch_size=args.eval_batch_size if args.eval_batch_size else 1,
        # stop_id_sequences=[[stop_sequnce]] 
    )


    if save_path:
        fout = open(save_path, "w")
    predictions = []
    for example in examples:
        sub_prediction = []
        for sub_example in example:
            prediction = outputs.pop(0)
            sub_example["prediction"] = prediction
            sub_prediction.append(prediction)
        predictions.append(sub_prediction)
    if save_path:
        fout.write(json.dumps(examples) + "\n")

    # test_gold_triggers, test_gold_events, test_pred_triggers, test_pred_events = get_trigger(examples)
    # test_scores = cal_scores(test_gold_triggers, test_pred_triggers, test_gold_events, test_pred_events)
    
    test_gold_triggers, test_pred_triggers = get_trigger(examples)
    test_scores = cal_scores(test_gold_triggers, test_pred_triggers)

    print("---------------------------------------------------------------------")
    print('Trigger I  - P: {:6.2f} ({:4d}/{:4d}), R: {:6.2f} ({:4d}/{:4d}), F: {:6.2f}'.format(
        test_scores['tri_id'][3] * 100.0, test_scores['tri_id'][2], test_scores['tri_id'][1], 
        test_scores['tri_id'][4] * 100.0, test_scores['tri_id'][2], test_scores['tri_id'][0], test_scores['tri_id'][5] * 100.0))
    print('Trigger C  - P: {:6.2f} ({:4d}/{:4d}), R: {:6.2f} ({:4d}/{:4d}), F: {:6.2f}'.format(
        test_scores['tri_cls'][3] * 100.0, test_scores['tri_cls'][2], test_scores['tri_cls'][1], 
        test_scores['tri_cls'][4] * 100.0, test_scores['tri_cls'][2], test_scores['tri_cls'][0], test_scores['tri_cls'][5] * 100.0))
    if save_path:
        save_score_path = save_path.replace(".jsonl", "_scores.txt")
        with open(save_score_path, "w") as F:
            F.write('Trigger I  - P: {:6.2f} ({:4d}/{:4d}), R: {:6.2f} ({:4d}/{:4d}), F: {:6.2f}\n'.format(
                test_scores['tri_id'][3] * 100.0, test_scores['tri_id'][2], test_scores['tri_id'][1], 
                test_scores['tri_id'][4] * 100.0, test_scores['tri_id'][2], test_scores['tri_id'][0], test_scores['tri_id'][5] * 100.0))
            F.write('Trigger C  - P: {:6.2f} ({:4d}/{:4d}), R: {:6.2f} ({:4d}/{:4d}), F: {:6.2f}\n'.format(
                test_scores['tri_cls'][3] * 100.0, test_scores['tri_cls'][2], test_scores['tri_cls'][1], 
                test_scores['tri_cls'][4] * 100.0, test_scores['tri_cls'][2], test_scores['tri_cls'][0], test_scores['tri_cls'][5] * 100.0))
# for example, output in zip(examples, outputs):
        # example["raw_output"] = output
        
        # # only keep the first part of the output - this is mainly for vanilla language models.
        # output = output.strip().split("\n\n")[0].strip()

        # # extract the first answer after `So the answer is` and before the next period.
        # # if there is no such answer, we will just use the raw output.
        # results = re.search(r"So the answer is (.*?)\.", output)
        # if results:
        #     prediction = results.group(1).strip()
        # else:
        #     prediction = output.strip()

        # example["prediction"] = output
        # predictions.append(output)
        # if save_path:
        #     fout.write(json.dumps(example) + "\n")        

    # assert len(predictions) == len(targets), "number of predictions and targets are not the same."
    # return predictions
    # return exact_match.compute(predictions=predictions, references=targets, ignore_case=True, ignore_punctuation=True)["exact_match"]


def eval_openai_chat_engine(args, examples, task_prompt, save_path=None):
    targets = [example["target"] for example in examples]
    instances = []
    for i, example in enumerate(examples):
        prompt = task_prompt.strip() + "\n\nQ: " + example["input"] + "\nA:"
        instances.append({
            "id": example["id"] if "id" in example else i,
            "prompt": prompt,
        })

    if save_path:
        openai_result_save_path = os.path.join(os.path.dirname(save_path), os.path.basename(save_path).split(".")[0] + "_openai_results.jsonl")
    
    results = query_openai_chat_model(
        engine=args.openai_engine,
        instances=instances,
        batch_size=args.eval_batch_size if args.eval_batch_size else 10,
        output_path=openai_result_save_path if save_path else None,
    )

    outputs = [result["output"] for result in results]
    assert len(outputs) == len(targets), "number of predictions and targets are not the same."

    if save_path:
        fout = open(save_path, "w")

    predictions = []
    for example, output in zip(examples, outputs):
        example["raw_output"] = output
        # extract the first answer after `So the answer is` and before the next period.
        # if there is no such answer, we will just use the raw output.
        results = re.search(r"So the answer is (.*?)\.", output)
        if results:
            prediction = results.group(1).strip()
        else:
            prediction = output.strip()
        example["prediction"] = prediction
        predictions.append(prediction)
        if save_path:
            fout.write(json.dumps(example) + "\n")        

    assert len(predictions) == len(targets), "number of predictions and targets are not the same."
    return exact_match.compute(predictions=predictions, references=targets, ignore_case=True, ignore_punctuation=True)["exact_match"]


def main(args):
    random.seed(42)

    all_tasks = {}
    # task_files = glob.glob(os.path.join(args.data_dir, "bbh", "*.json"))
    task_files = [
        os.path.join(args.data_dir, args.test_file),
        os.path.join(args.data_dir, args.valid_file),
    ]
    for task_file in tqdm.tqdm(task_files, desc="Loading tasks"):
        data = []
        with open(task_file, "r") as f:
            task_name = os.path.basename(task_file).split(".")[0]
            
            for line in f.readlines():
                data.append(json.loads(line))
            
            all_tasks[task_name] = data
            if args.max_num_examples_per_task:
                all_tasks[task_name] = random.sample(all_tasks[task_name], args.max_num_examples_per_task)

    all_prompts = {}
    # cot_prompt_files = glob.glob(os.path.join(args.data_dir, "cot-prompts", "*.txt"))
    # for cot_prompt_file in tqdm.tqdm(cot_prompt_files, desc="Loading prompts"):
    #     with open(cot_prompt_file, "r") as f:
    #         task_name = os.path.basename(cot_prompt_file).split(".")[0]
    #         task_prompt = "".join(f.readlines()[2:])
    #         if args.no_cot:
    #             prompt_fields = task_prompt.split("\n\n")
    #             new_prompt_fields = []
    #             for prompt_field in prompt_fields:
    #                 if prompt_field.startswith("Q:"):
    #                     assert "So the answer is" in prompt_field, f"`So the answer is` not found in prompt field of {task_name}.txt."
    #                     assert "\nA:" in prompt_field, "`\nA:` not found in prompt field."
    #                     answer = prompt_field.split("So the trigger is")[-1].strip()
    #                     question = prompt_field.split("\nA:")[0].strip()
    #                     new_prompt_fields.append(question + "\nA: " + answer)
    #                 else:
    #                     new_prompt_fields.append(prompt_field)
    #             task_prompt = "\n\n".join(new_prompt_fields)
    for task_name in all_tasks.keys():
        all_prompts[task_name] = "\n\n So what is the trigger?"

    assert set(all_tasks.keys()) == set(all_prompts.keys()), "task names in task data and task prompts are not the same."

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "predictions"), exist_ok=True)

    if args.model_name_or_path:
        print("Loading model and tokenizer...")
        model, tokenizer = load_hf_lm_and_tokenizer(
            model_name_or_path=args.model_name_or_path, 
            tokenizer_name_or_path=args.tokenizer_name_or_path, 
            load_in_8bit=args.load_in_8bit, 
            # Sometimes it will make model predict unknown
            load_in_half=True,
            gptq_model=args.gptq
        )

    special_tokens = ['<trigger>', '<sep>', '<Trigger>']
    tokenizer.add_tokens(special_tokens)


    performance = {}
    for task_name in tqdm.tqdm(all_tasks.keys(), desc="Evaluating"):
        task_examples = all_tasks[task_name]
        prompt = all_prompts[task_name]

        if args.model_name_or_path:
            task_perf = eval_hf_model(
                args, 
                model, 
                tokenizer, 
                task_examples, 
                prompt, 
                save_path=os.path.join(args.save_dir, "predictions", f"{task_name}.jsonl")
            )
        else:
            task_perf = eval_openai_chat_engine(
                args,
                task_examples,
                prompt,
                save_path=os.path.join(args.save_dir, "predictions", f"{task_name}.jsonl")
            )
            
    # print("---------------------------------------------------------------------")
    # print('Trigger I  - P: {:6.2f} ({:4d}/{:4d}), R: {:6.2f} ({:4d}/{:4d}), F: {:6.2f}'.format(
    #     test_scores['tri_id'][3] * 100.0, test_scores['tri_id'][2], test_scores['tri_id'][1], 
    #     test_scores['tri_id'][4] * 100.0, test_scores['tri_id'][2], test_scores['tri_id'][0], test_scores['tri_id'][5] * 100.0))
    # print('Trigger C  - P: {:6.2f} ({:4d}/{:4d}), R: {:6.2f} ({:4d}/{:4d}), F: {:6.2f}'.format(
    #     test_scores['tri_cls'][3] * 100.0, test_scores['tri_cls'][2], test_scores['tri_cls'][1], 
    #     test_scores['tri_cls'][4] * 100.0, test_scores['tri_cls'][2], test_scores['tri_cls'][0], test_scores['tri_cls'][5] * 100.0))
    #     performance[task_name] = task_perf
    #     print(f"Task {task_name} - EM: {task_perf}")

    # with open(os.path.join(args.save_dir, "metrics.json"), "w") as fout:
    #     performance["average_exact_match"] = sum(performance.values()) / len(performance)
    #     print(f"Average EM: {performance['average_exact_match']}")
    #     json.dump(performance, fout, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/ace")
    parser.add_argument("--valid_file", type=str, default="")
    parser.add_argument("--test_file", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="")
    parser.add_argument("--model_name_or_path", type=str, default=None, help="if specified, we will load the model to generate the predictions.")
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None, help="if specified, we will load the tokenizer from here.")
    # parser.add_argument("--openai_engine", type=str, default=None, help="if specified, we will use the OpenAI API to generate the predictions.")
    # parser.add_argument("--no_cot", action="store_true", help="if specified, chain of thoughts will be removed from the prompts.")
    parser.add_argument("--max_num_examples_per_task", type=int, default=None, help="maximum number of examples to evaluate per task.")
    parser.add_argument("--eval_batch_size", type=int, default=1, help="batch size for evaluation.")
    parser.add_argument("--load_in_8bit", action="store_true", help="load model in 8bit mode, which will reduce memory and speed up inference.")
    parser.add_argument("--gptq", action="store_true", help="If given, we're evaluating a 4-bit quantized GPTQ model.")
    parser.add_argument("--use_chat_format", action="store_true", help="If given, the prompt will be encoded as a chat format with the roles in prompt.")
    args = parser.parse_args()

    # model_name_or_path and openai_engine cannot be both None or both not None.
    # assert (args.model_name_or_path is None) != (args.openai_engine is None), "Either model_name_or_path or openai_engine should be specified."
    main(args)
