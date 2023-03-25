import json
import random
import csv


formats_dict = {
	"aqua": [
        ("Q: {question} Let's give some random thoughts before answering.",
         "{chain_of_thought}\nTherefore, the answer is {answer}."),
        ("{question} Hmmm, my stream of consciousness:",
         "{chain_of_thought} So, the answer is {answer}."),
        ("Give a quick stream of consciousness before answering the following "
         "question. {question}", "{chain_of_thought}\nThe answer: {answer}."),
        ("Use some thinking to answer the following question. {question}",
         "{chain_of_thought} So the final answer is {answer}."),
        ("Student: {question}.\nAnother student: Let's say, hmmm...\n",
         "{chain_of_thought} Final answer: {answer}."),
        ("{question} Think first, then make a decision. Some random thoughts:",
         "{chain_of_thought} So the answer is {answer}."),
        ("{question} Now, let's think a bit. Some random thoughts:",
         "{chain_of_thought}\nThe final answer: {answer}."),
        ("{question} Stream of consciousness:",
         "{chain_of_thought}\nThus, the answer is {answer}."),
        ("Question: {question} Random thoughts:",
         "{chain_of_thought} The final answer: {answer}."),
        ("{question} OK. Let's think. Some random thoughts first:",
         "{chain_of_thought} The answer: {answer}."),
        ("Give stream of consciousness and then the final answer. {question}",
         "{chain_of_thought}\nThe final answer: {answer}."),
        ("{question} Stream of consciousness first, then make a decision:",
         "{chain_of_thought}\nThus, the answer is {answer}."),
        ("Question: {question} Let's think first. Some random reasoning:",
         "{chain_of_thought} The final answer: {answer}."),
        ("Some question: {question}\nSome stream of consciousness:",
         "{chain_of_thought} The answer: {answer}."),
        ("{question} Let's think first. Stream of consciousness:",
         "{chain_of_thought}\nSo, the answer is {answer}."),
    ],    
    	"creak": [
        ("Given the following question, let's solve step-by-step. {question}\n",
         "{chain_of_thought}\nThe final answer: {answer}."),
        ("My question: {question}\nPlease think gradually:",
         "{chain_of_thought}\nThus, the answer is {answer}."),
        ("Give the rationale and then the answer. {question}",
         "{chain_of_thought} The final answer: {answer}."),
        ("Q: {question}\nChain-of-thought:",
         "{chain_of_thought} The answer: {answer}."),
        ("{question}\nChain of thought and solution for this question is:",
         "{chain_of_thought}\nSo, the answer is {answer}."),
        ("Question: {question} Let's think first. Step-by-step reasoning:",
         "{chain_of_thought}\nSo, the final answer is {answer}."),
        ("{question}\nYour chain-of-thought:",
         "{chain_of_thought} The answer is {answer}."),
        ("{question} Step-by-step reasoning process:",
         "{chain_of_thought} Therefore, the final answer is {answer}."),
        ("{question} The thought process:",
         "{chain_of_thought}\nTherefore, the answer is {answer}."),
        ("{question} Let's think first. Step-by-step reasoning process:",
         "{chain_of_thought} So, the answer is {answer}."),
    ],
        "ecqa": [
        ("{question}\nPlease answer and provide answer explanation.",
         "{chain_of_thought} The final answer: {answer}."),
        ("{question}\nStep-by-step reasoning process below:\n",
         "{chain_of_thought} The answer: {answer}."),
        ("{question} Hmmm, let me think.",
         "{chain_of_thought}\nSo, the answer is {answer}."),
        ("{question}\nLet's think now! Step-by-step reasoning:",
         "{chain_of_thought}\nSo, the final answer is {answer}."),
        ("next question: {question}\nreasoning:",
         "{chain_of_thought} The answer is {answer}."),
        ("Use reasoning to lead to the answer of the following question:\n"
         "{question}\n Reasoning process:",
         "{chain_of_thought} Therefore, the final answer is {answer}."),
        ("{question} Let's give stream of consciousness first:",
         "{chain_of_thought}\nTherefore, the answer is {answer}."),
        ("{question} Let's think step by step:",
         "{chain_of_thought} So, the answer is {answer}."),
        ("I'll give you a question, please answer with step-by-step reasoning "
         "process. {question}\n", "{chain_of_thought}\nThe answer: {answer}."),
        ("{question}\nLet's think carefully first. Step-by-step reasoning "
         "process:", "{chain_of_thought} So the final answer is {answer}."),
    ],
        "esnli": [
        ("{question}\nLet's solve step-by-step:",
         "{chain_of_thought} The answer is {answer}."),
        ("{question} Step by step answer:",
         "{chain_of_thought} Therefore, the final answer is {answer}."),
        ("{question} Stream of thoughts:",
         "{chain_of_thought}\nTherefore, the answer is {answer}."),
        ("{question} Now, let's be accurate as possible. Some thinking first:",
         "{chain_of_thought} So, the answer is {answer}."),
        ("Denny asked: {question}.\nLe: OK, so how can I answer with some "
         "explanation?\n", "{chain_of_thought}\nThe answer: {answer}."),
        ("Student: {question}.\nTeacher: Let's think:\n",
         "{chain_of_thought} So the final answer is {answer}."),
        ("{question} Let's be accurate as possible and think first.",
         "{chain_of_thought} Final answer: {answer}."),
        ("Please answer the following question by reasoning step-by-step. "
         "{question}. Step-by-step reasoning:",
         "{chain_of_thought} So the answer is {answer}."),
        ("{question} A step-by-step solution is:\n",
         "{chain_of_thought}\nThe final answer: {answer}."),
        ("Leo: {question}\nMei: OK, So, let's think first...\nMe:",
         "{chain_of_thought}\nThus, the answer is {answer}."),
    ],
        "gsm8k": [
        ("{question} Let's think first. Chain of thought:",
         "{chain_of_thought}\nTherefore, the answer is {answer}."),
        ("{question} Think carefully first, then make a decision:",
         "{chain_of_thought} So, the answer is {answer}."),
        ("{question} Let's be accurate as possible.",
         "{chain_of_thought}\nThe answer: {answer}."),
        ("{question} Give me reasons, before answering the question",
         "{chain_of_thought} So the final answer is {answer}."),
        ("Lizzy: {question}.\nMe: Hmmm, let me think. I think this is the "
         "detailed solution:", "{chain_of_thought} Final answer: {answer}."),
        ("Question: {question} Think carefully first, then make a decision:",
         "{chain_of_thought} So the answer is {answer}."),
        ("Give the step-by-step reasoning process and then the final answer. "
         "{question}", "{chain_of_thought}\nThe final answer: {answer}."),
        ("{question}\nThoughts? Step-by-step reasoning:",
         "{chain_of_thought}\nThus, the answer is {answer}."),
        ("My question is: {question} Your thoughts:",
         "{chain_of_thought} The final answer: {answer}."),
        ("{question} Let's answer step by step:",
         "{chain_of_thought} The answer: {answer}."),
    ],
        "qasc": [
        ("Question: {question}\nLet's be accurate as possible and think "
         "step-by-step.", "{chain_of_thought}\nThe answer: {answer}."),
        ("{question} Let's solve this problem gradually.\n",
         "{chain_of_thought} So the final answer is {answer}."),
        ("Question to you: {question}.\nLet's reason step-by-step:",
         "{chain_of_thought} Final answer: {answer}."),
        ("{question} Think carefully first, then make a decision. My thoughts:",
         "{chain_of_thought} So the answer is {answer}."),
        ("{question} Let's be accurate as possible.",
         "{chain_of_thought}\nThe final answer: {answer}."),
        ("Q: {question}\nLet's think step by step below.\n",
         "{chain_of_thought}\nThus, the answer is {answer}."),
        ("Let's think step by step! {question}\nThe thinking starts now:",
         "{chain_of_thought} The final answer: {answer}."),
        ("{question}\nHmmm, let me think. I don't want to be wrong, so I got "
         "to be careful.", "{chain_of_thought} The answer: {answer}."),
        ("Use reasoning to answer the following question. {question}",
         "{chain_of_thought}\nSo, the answer is {answer}."),
        ("{question} OK. Let's think hard:",
         "{chain_of_thought}\nSo, the final answer is {answer}."),
    ],
        "qed": [
        ("{question}\nSteam of consciousness below:\n",
         "{chain_of_thought}\nSo, the answer is {answer}."),
        ("{question} Let's give stream of consciousness first:",
         "{chain_of_thought}\nSo, the final answer is {answer}."),
        ("Quoc: {question}\nHW Chung: OK, some thoughts:",
         "{chain_of_thought} The answer is {answer}."),
        ("Q: {question} Let's give stream of consciousness first:",
         "{chain_of_thought} Therefore, the final answer is {answer}."),
        ("I got a question for you: {question}\nLet's think first:",
         "{chain_of_thought}\nTherefore, the answer is {answer}."),
        ("{question} Okie... think carefully first, then make a decision:",
         "{chain_of_thought} So, the answer is {answer}."),
        ("Output a stream of consciousness before answering the following. "
         "{question}", "{chain_of_thought}\nThe answer: {answer}."),
        ("{question} Let's think fast. Stream of consciousness:",
         "{chain_of_thought} So the final answer is {answer}."),
        ("Use stream of consciousness to answer the following. {question}",
         "{chain_of_thought} Final answer: {answer}."),
        ("Q: {question}\nLet's give stream of consciousness below\n",
         "{chain_of_thought} So the answer is {answer}."),
        ("Give a stream of consciousness and then the final answer. {question}",
         "{chain_of_thought}\nSo, the final answer is {answer}."),
        ("{question} OK. Let's think. My stream of consciousness:",
         "{chain_of_thought} The answer is {answer}."),
        ("Answer the following Q with stream of consciousness. {question}",
         "{chain_of_thought} Therefore, the final answer is {answer}."),
        ("Give some stream of consciousness and then the answer. {question}",
         "{chain_of_thought}\nTherefore, the answer is {answer}."),
        ("{question} Let's have some stream of consciousness first.",
         "{chain_of_thought} So, the answer is {answer}."),
    ],
        "sensemaking": [
        ("{question} Let's reason step by step:",
         "{chain_of_thought} Final answer: {answer}."),
        ("Question: {question}\nPlease answer this question gradually:",
         "{chain_of_thought} So the answer is {answer}."),
        ("See question below:\n{question}\nReason slowly and give your answer.",
         "{chain_of_thought}\nThe final answer: {answer}."),
        ("OK. You'll be given the following question. Please do "
         "chain-of-thought reasoning.\n{question}",
         "{chain_of_thought}\nThus, the answer is {answer}."),
        ("{question} Let's be accurate as possible. So think first.",
         "{chain_of_thought}\nThe final answer: {answer}."),
        ("Q: {question}\nLet's solve this gradually.\n",
         "{chain_of_thought} The answer is {answer}."),
        ("Let's think step by step! {question}\n",
         "{chain_of_thought} The answer: {answer}."),
        ("{question}\nHmmm, let me think. I want to lay out the solution "
         "in details.", "{chain_of_thought} The answer is {answer}."),
        ("Answer the following question, with explanation first. {question}",
         "{chain_of_thought}\nSo, the answer is {answer}."),
        ("{question} Let me think hard. Detailed solution:",
         "{chain_of_thought}\nThe answer is {answer}."),
    ],
        "strategyqa": [
        ("{question}\nThink slowly and carefully, before giving your answer.",
         "{chain_of_thought}\nSo, the answer is {answer}."),
        ("{question} Please answer step by step:",
         "{chain_of_thought}\nSo, the final answer is {answer}."),
        ("{question}\nChain of thought:",
         "{chain_of_thought} The answer is {answer}."),
        ("Answer the following question by reasoning step-by-step. {question}",
         "{chain_of_thought} Therefore, the final answer is {answer}."),
        ("{question} Given the above question, please answer with reasoning "
         "first!", "{chain_of_thought}\nTherefore, the answer is {answer}."),
        ("{question} Think carefully first, then make a decision:",
         "{chain_of_thought} So, the answer is {answer}."),
        ("Q: {question} Now, let's think step by step:",
         "{chain_of_thought}\nThe answer: {answer}."),
        ("Answer the following question, but give the rationale first. "
         "{question}", "{chain_of_thought} So the final answer is {answer}."),
        ("{question} Hmmm, my chain of thoughts:",
         "{chain_of_thought} Final answer: {answer}."),
        ("Let's answer this question slowly: {question}\n",
         "{chain_of_thought} So the answer is {answer}."),
    ],}



def cot_process(data):
    final_list=[]
    x=0
    with open('./'+data+'_train.tsv') as f:
        tsvreader = csv.reader(f, delimiter='\t')
        for line in tsvreader:
            assert len(line) == 3
            x+=1
            if x%500==0:
                print(x)
            new_dict={}
            question_option_str = line[0]

            answer_str = line[1]
            cot_str = line[2]

            which_tem = random.randint(0,len(formats_dict[data])-1)
            template =  formats_dict[data][which_tem]
            assert len(template) == 2

            new_dict={}
            tem_instruction = template[0].format(question=question_option_str)
            tem_input = ""
            tem_output = template[1].format(chain_of_thought=cot_str,answer=answer_str)

          
            new_dict["instruction"] = tem_instruction
            new_dict["input"] = tem_input
            new_dict["output"] = tem_output

            final_list.append(new_dict)

    fw=open("../formated_cot_data/"+data+"_train.json", "w", encoding="utf8")
    json.dump(final_list, fw, ensure_ascii=False)

cot_process(data="strategyqa")


