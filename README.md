## Reproducibility Study of [Cooperation, Competition, and Maliciousness: LLM-Stakeholders Interactive Negotiation](https://arxiv.org/abs/2309.17234)

## Abstract 
This paper presents a reproducibility study and extension of "Cooperation, Competition, and Maliciousness: LLM-Stakeholders Interactive Negotiation." We validate the original findings using a range of open-weight models (1.5B-70B parameters), GPT-4, and GPT-4o Mini while introducing several novel contributions. We analyze the Pareto front of the games, propose a communication-free baseline to test whether successful negotiations are possible without agent interaction, evaluate recent small language models' performance, analyze structural information leakage in model responses, and implement an inequality metric to assess negotiation fairness. Our results demonstrate that smaller models (<10B parameters) struggle with format adherence and coherent responses, but larger open-weight models can approach proprietary model performance. Additionally, in many scenarios, single-agent approaches can achieve comparable results to multi-agent negotiations, challenging assumptions about the necessity of agent communication to perform well on the benchmark. This work also provides insights into accessibility, fairness, environmental impact, and privacy considerations of LLM-based negotiation systems.

## Our Modifications and Contributions
- Originally, the authors set the model temperature to 0 and a random order of the agents for each round. However, the authors did not add a fixed seed in the code, making their exact results not reproducible. To address this, we added fixed seeds to the code setup. We then ran all the experiments 10 times with seeds ranging from 1 to 10. We also corrected the `do_sample` parameter in the Hugging Face pipeline, changing it from `True` to `False` to set up the correct greedy decoding configuration.

  Although the original paper performed 20 runs, we reduced the number of experiments due to computational constraints. This decision was also influenced by the fact that the exact reproduction of the results is not possible.

- The authors' code uses fixed `float32` precision for models loaded from Hugging Face, which can be inefficient and affect the generalization for models like Phi-4. We resolved this issue by using `auto` precision, which allows the framework to automatically select the most appropriate precision for the available hardware and model requirements. Additionally, we manually adjusted the precision of Llama models to `float16` precision. This configuration was recommended for managing memory constraints while maintaining an acceptable performance. It is important to note that `float16` is not a form of quantization but a more efficient memory format.

- In their prompts, authors ask models to format their answers using specific tags, such as `<DEAL>`, `<PLAN>`, `<SCRATCHPAD>`, and `<ANSWER>`. These tags are later used to break the model's answer into the Chain-of-Thought structure. Models generally comply with these requirements. However, during evaluation, the `<DEAL>` tags are not used for parsing; instead, the suggested deals are parsed directly from the models' public answers. This approach often leads to unparsable deals, negatively impacting the scores of certain models.

  We consider this a bug because models sometimes respond in a more human-like manner, describing the deals' meanings (e.g., "a loan of 200 dollars" instead of "option A2"). Additionally, the prompts are misleading, implying that the placement of the `<DEAL>` tag within the answer does not matter. This inconsistency skews the evaluation scores. We have addressed this issue and now report the corrected scores.

- We found a bug in the game evaluation: the number of acceptable deals was computed incorrectly, leading to incorrect levels of difficulty and game analysis in the 'Turning the game difficulty' section of the original paper.

- We found a bug in the evaluation notebook: the code inconsistently evaluated if the deal was acceptable. In one part of the code, the score was required to be greater than the minimal threshold, but in the rest of the code, a score equal to the minimal threshold was considered acceptable. After fixing the bug, a few model evaluations changed.

- We also fixed minor bugs, such as bracket mismatches, setting the value of the API key, faulty conditional parses, and the missing `matplotlib` dependency.


**From here onwards, we preserve the README from the original paper because we believe it was well written and formatted.**

### Example 
- You can find [here](https://amrgomaaelhady.github.io/LLM-Deliberation-Demo/) an example of one of our runs with GPT-4 

<p align="center">
<img src="https://github.com/S-Abdelnabi/LLM-Deliberation/blob/main/teaser.png" width="750">
</p>

---

## The repo includes:

* All games and game variants developed in the paper 
* All logs from experiments in the paper
* Code to run simulations
* Evaluation code
* Guide on how to adjust the experiments and extend the setup for other scenarios. 

---

## Table of Content 
- [Setup](#setup)
- [Games](#Games)
- [Setting the game and simulation configuration](#Setting-the-game-and-simulation-configuration)
- [Guide on how the prompts are organized](#Guide-on-how-the-prompts-are-organized)
- [Running the simulation](#Running-the-simulation)
- [Evaluation](#Evaluation)
- [Logs](#Logs)
- [Citation](#citation)

---
  

## Setup 

- Create a new enviroment and install the following:
```
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install conda-forge::transformers
pip install google-cloud-aiplatform
pip install openai
pip install accelerate
```

---

## Games 
- All games can be found under `games_descriptions. Current games are:
  - `Base` game
  - `Base rewritten`: base game rewritten by GPT-4
  - 7-player and 6-issue variant extended from base game
  - New games created by prompting GPT-4 and manual curation (`game1`, `game2`, `game3`)
    
- Games are organized as follows:
  - `global_instructions.txt`:
    - These are global instructions about project and issues given to all agents.
    - Name of agents should be put between quotations `""` and be consistent all along the global instructions (will be parsed when creating initial prompts, more details about that later)
      
  - `<GAME>/individual_instructions`:
    - There should be sub-folders that correspond to the `incentive` of the player (e.g., cooperative). I.e., `<GAME>/individual_instructions/<INCENTIVE>`
    - Under the sub-folders, there should be files that correspond to players. 
    - These are the confidential information given to agents about their preferences in addition to any agent-specific instructions about their incentives (e.g., `you need to maximize this option as much as possible`)
    - The files contain the scores as placeholders that will be populated when forming the initial prompt (more details about that later).
  - `<GAME>/scores_files`:
    - These are files that contain the scores of players for each issue 
    - Each line is a comma-separated line of scores per issue
    - Line 1 corresponds to issue A, etc.
    - The last line is the minimum threshold for the agent.
    - If you just want to change scores or thresholds without affecting the overall game and priorities, just change values in the scores files without changing the other files.
      
  - `<GAME>/initial_deal.txt`: This is the most preferred option for `p1` that will be used to start the negotiation.
    
- We include `greedy`, `targeted_adv`, `untargeted_adv`, `cooperative` incentives for the `base` game according to the results in the paper. Other games have currently only the `cooperative` variant.
- **If you would like to support another incentive**, create a new sub-directory and write the individual instructions for agents you would like to combine that incentive with. 

---

## Setting the game and simulation configuration 
- Change `<GAME>/config.txt` to run customized combinations of agents' models, incentives, etc and varying number of agents, etc.
- Each line in `config.txt` corresponds to one agent.
- Each line should be organized as `<AGENT NAME>, <FILE NAME>, <ROLE>, <INCENTIVE>, <MODEL>`:
  - `<AGENT NAME>` this is the agent's game name as written in the `global_instructions.txt` file.
  - `<FILE NAME>` this is the agent's file name under `<GAME>/individual_instructions` and `<GAME>/scores_files`
  - `<ROLE>` a specific role for the agent. At the moment this can be `p1`, `p2`, `target` (for the target agent in `targeted_adv` incentive), or `player` (default for all others).
  - `<INCENTIVE>` the incentive for the agent. This can be `greedy`, `targeted_adv`, `untargeted_adv`, or `cooperative`. A sub-directory of the same name must be included under `<GAME>/individual_instructions`.
  - <MODEL> the model that will be used for this agent. You can specify different models for different agents. The code now supports *GPT* models via *Azure APIs* or *OpenAI APIs*, *Gemini*, or Hugging Face models. **For Hugging Face models, write `hf_<MODEL>`**.
- If you would like to run the same game but with fewer agents, remove that agent's line from the config file.

---

## Guide on how the prompts are organized 
- The agents have 1) **initial prompts** and 2) **round prompts**. They are formed as follows:

1- **initial prompts**
- `initial_prompts.txt` first reads the *global instructions* and replaces the agent's name with (``<AGENT_NAME> (represented by you``).
```python
self.global_instructions = self.load_global_instructions(os.path.join(game_description_dir,'global_instructions.txt'))
```
- Next, scores and indiviual instructions are read and combined:
```python
individual_scores_file = os.path.join(game_description_dir,'scores_files', agent_file_name+'.txt')
self.scores = self.load_scores(individual_scores_file)
        
individual_instructions_file = os.path.join(game_description_dir,'individual_instructions',incentive, agent_file_name+'.txt')
self.individual_instructions = self.load_individual_instructions(individual_instructions_file)
```
- The initial prompt contains **scoring** and **voting** instructions that are given the same to all agents (E.g., who is `p1`, the round schema, etc.).
- There also specific **incentive** instructions (e.g., for cooperative, it includes something like `any deal with a score higher than your minimum threshold is preferable to you than no deal. You are very open to any compromise to achieve that`).
- The final initial prompt is:
```python
final_initial_prompt = self.global_instructions + '\n' + self.individual_instructions +  scoring_rules + voting_rules + incentive_rules
```
- `InitialPrompt` class supports changing the number of agents and the number of classes and it takes `p1` and `p2` from `main.py` (more details later). It also call the incentive-specific functions based on the agents' incentives defined in the `config.txt` files. 

2- **round prompts**
- The rounds' prompts get appended to the initial prompts at each interaction.
- Round prompts are constructed as:
```python
slot_prompt = history_prompt + scratch_pad + unified_instructions + plan_prompt 
```
- `history_prompt` is the n-window history of the negotiation of **public answers**. They are formatted such that the agent's name is replaced by `You: `.
- `scratch_pad` is instructions on the individual CoT steps along with incentive-related instructions of the goals. Currently, each `incentive` has a scratch pad function that gets called based on the agent's incentive.
- `unified_instructions` are instructions on how to format answers.
- `plan_prompt` are instructions on how to form plans (won't be called for the last time the agent is prompted).

### Supporting new incentives:
- If you would like to support another incentive, create new functions for that incentive in *initial* and *round* prompts if needed. 

---

## Running the simulation 

- After changing `config.txt`, run the simulation as:
  
```
python main.py --exp_name <OUTPUT_DIR> --agents_num <NUM> --issues_num <NUM> --window_size <NUM> --game_dir ./games_descriptions/<GAME> --rounds_num <NUM>
```
- If you need to run Azure APIs, run with the flag `--azure``
- Specify API keys
- Change the number of agents and issues according to the game.
- We used `rounds_num` as (`4*agents_num`)
- The training script will create an output dir with `exp_name` under `./games_descriptions/<GAME>`. It will also copy `config.txt` and `<GAME>/scores_files`
- The history file will have the following format:
```python
history['content']["rounds"].append({'agent':agent_name, 'prompt': prompt, 'full_answer': full_answer, 'public_answer': public_answer})
```
  - `rounds` is a list of length (`args.rounds_num` + 2). The first is the initial prompts and the last one is the deal suggestion by `p1`. `prompt` is the prompt given at this round. `full_answer` is the full answer including the CoT. `public_answer` is the extracted public answer given to agents in the history. 

---

## Evaluation 
1- `evaluation/evaluate_deals.ipynb`:
- Measures metrics: any success rate, final success rate, and ratio of wrong scores. Change the following according to the game:
  ```python
  HOME = '<HOME>'
  OUTPUT_DIR = os.path.join(HOME,'LLM-Deliberation/games_descriptions/base/output/all_coop')
  AGENTS_NUM = 6
  ISSUES_NUM = 5
  NUM_ROUNDS = 24
  ```
- Use the same notebook to create figures of agents' deals such as the ones in the paper:
<p align="center">
<img src="https://github.com/S-Abdelnabi/LLM-Deliberation/blob/main/p1.png" width="350">
</p>

2- `evaluation/score_leakage.py`:
- Use GPT-4 as a judge to evaluate whether scores where leaked in the public answers.
- Specify the following arguments:
```python
MAX_THREADS = 60 

parser = argparse.ArgumentParser(
                    prog='Verifier')

parser.add_argument('--azure_openai_api', default='', help='azure api') 
parser.add_argument('--azure_openai_endpoint', default='', help='azure endpoint')   
parser.add_argument('--model_name', default='', help='azure model')  
parser.add_argument('--exp_dir')

args, _ = parser.parse_known_args()

os.environ["AZURE_OPENAI_API_KEY"] = args.azure_openai_api
os.environ["AZURE_OPENAI_ENDPOINT"] = args.azure_openai_endpoint
```
- Note that this script creates parallel calls to GPT-4. **Be mindful of cost as it may accumulate quickly**.

3- `evaluation/adjust_games.ipynb` 
- This script can be used to visualize the number of possible deals (and also possible deals per agent) after changing the scores or minimum thresholds of agents.
- Change the following parameters:
```python
HOME = '/HOME/'
GAME_DIR = os.path.join(HOME,'LLM-Deliberation/games_descriptions/base/')
AGENTS_NUM = 6
ISSUES_NUM = 5
```

---

## Logs 

- We share logs of most of our experiments under `logs`.
- Please note that some logs were generated with our previous code base and the logs were saved in a slightly different scheme. Please refer to `old_code` branch (we will add more details TBD).

---

