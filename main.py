import os
import torch
import shutil
import random
import argparse
import numpy as np

from utils.agent import Agent 
from utils.initial_prompts import InitialPrompt
from utils.rounds import RoundPrompts
from utils.save_utils import create_outfiles, save_conversation 
from utils.utils import load_setup, set_constants, randomize_agents_order, setup_hf_model


##### SET ARGUMENTS AND OUTPUT DIRECTORY #####
parser = argparse.ArgumentParser(description='big negotiation!!')

parser.add_argument('--temp', type=float, default='0')
parser.add_argument('--agents_num', type=int, default=6)
parser.add_argument('--issues_num', type=int, default=5)
parser.add_argument('--rounds_num', type=int, default=24)
parser.add_argument('--window_size', type=int, default=6)
parser.add_argument('--output_dir', type=str, default='./output/')
parser.add_argument('--game_dir', type=str, default='./games_descriptions/base')
parser.add_argument('--exp_name', type=str, default='all_coop')
parser.add_argument('--game_type', choices=["row1", "row2", "row3", "row4", "row5", "row6"], default="row5")
parser.add_argument('--single_agent', action='store_true')

#if restart, specifiy output_file to continue on 
parser.add_argument('--restart', action='store_true')
parser.add_argument('--output_file', type=str, default='history.json')

#if any gemini model, set this true 
parser.add_argument('--gemini', action='store_true')
parser.add_argument('--gemini_project_name', type=str, default='')
parser.add_argument('--gemini_loc', type=str, default='')
parser.add_argument('--gemini_model', type=str, default='gemini-1.0-pro-001')

#if any open-source model, set this true 
parser.add_argument('--hf_home', type=str, default='/disk1/')

#for GPTs and using Azure APIs, set this true 
parser.add_argument('--azure', action='store_true')
parser.add_argument('--azure_openai_api', default='', help='azure api') 
parser.add_argument('--azure_openai_endpoint', default='', help='azure endpoint')   

#for GPTs and OpenAI APIs, set key 
parser.add_argument('--api_key', type=str, default='', help='OpenAI key, set if using OpenAI APIs')

#dry run (instead of inferencing, just returns '<DRYRUN></DRYRUN>')
parser.add_argument('--dry_run', action='store_true', help='Do not inference models. Appends _dryrun to args.exp_name')
parser.add_argument('--dry_run_on_history', action='store_true')

#fixed seed for reproducibility
parser.add_argument('--seed', type=str, default='random', help='Random seed for reproducibility, used to shuffle agent order')

args = parser.parse_args()

SEED = int(args.seed) if args.seed != 'random' else random.randint(10000, 99999)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

OUTPUT_DIR = os.path.join(args.game_dir, args.output_dir, args.exp_name, args.game_type + ('_dryrun' if args.dry_run else ''))

# SET AZURE, OpenAI and GEMINI APIs env variables 
set_constants(args)

# Create output file, or load files if restart is given to continue on last experiments 
agent_round_assignment, start_round_idx, history  = create_outfiles(args, OUTPUT_DIR)

# Dump config file and scores in OUTPUT_DIR
shutil.copyfile(os.path.join(args.game_dir, 'config.txt'), os.path.join(OUTPUT_DIR, 'config.txt'))
shutil.copytree(os.path.join(args.game_dir, 'scores_files'), os.path.join(OUTPUT_DIR, 'scores_files'), dirs_exist_ok=True)

# Load setups of agents from config file. File should contain names, file names, roles, incentives, and models 
# Also load initial deal file and return a dict of role to agent names 
agents, initial_deal, role_to_agent_names = load_setup(args.game_dir, args.agents_num)

# Load HF models 
hf_models = {}

# Instaniate agents (initial prompt, round prompt, agent class)
for name in agents.keys(): 

    if 'hf' in agents[name]['model'] and not agents[name]['model'] in hf_models and not args.dry_run:
        hf_models[agents[name]['model']] = setup_hf_model(agents[name]['model'].split('hf_')[-1], cache_dir=args.hf_home, max_new_tokens=7000, dry_run_on_history=args.dry_run_on_history)
        print('hf model loaded')
        
    inital_prompt_agent = InitialPrompt(
        args.game_dir, 
        name, 
        agents[name]['file_name'],
        role_to_agent_names['p1'], 
        role_to_agent_names['p2'],
        num_issues=args.issues_num, 
        num_agents= args.agents_num, 
        incentive=agents[name]['incentive']
    )
    
    round_prompt_agent = RoundPrompts(
        name, 
        role_to_agent_names['p1'],
        initial_deal,
        incentive=agents[name]['incentive'], 
        window_size=args.window_size,
        target_agent=role_to_agent_names.get('target',''),
        rounds_num=args.rounds_num, 
        agents_num=args.agents_num,
        game_type=args.game_type
    )
        
    agent_instance = Agent(
        inital_prompt_agent,
        round_prompt_agent,
        name,
        args.temp,
        model=agents[name]['model'],
        model_seed=SEED,
        azure=args.azure,
        hf_models=hf_models,
        dry_run=args.dry_run,
        dry_run_on_history=iter(history['content']['rounds']) if args.dry_run_on_history else False,
    )

    agents[name]['instance'] = agent_instance

# If not restart, agent_round_assignment is empty, then randomize order 
if not args.restart and not args.dry_run_on_history: 
    agent_round_assignment = randomize_agents_order(agents, role_to_agent_names['p1'], args.rounds_num, args.single_agent)
if not args.restart and args.dry_run_on_history: 
    start_round_idx = 0

n_input_tokens = []
n_output_tokens = []

for round_idx in range(start_round_idx, args.rounds_num):
    if args.single_agent:
        current_agent = agent_round_assignment[round_idx]
        slot_prompt, agent_response = agents[current_agent]['instance'].execute_round(history['content'], round_idx)
        if not args.dry_run_on_history:
            if round_idx == 0:
                history = save_conversation(history, current_agent, agent_response, slot_prompt, agents=agents, round_assign=agent_round_assignment, initial=True)
            else:
                history = save_conversation(history, current_agent, agent_response, slot_prompt)
        else:
            n_input_tokens.append(agents[current_agent]['instance'].n_input_tokens[-1])
            n_output_tokens.append(agents[current_agent]['instance'].n_output_tokens[-1])
    else:
        if round_idx == 0:
            #For first round, initialize with p1 suggesting the first deal from 'initial_deal.txt' file 
            current_agent = role_to_agent_names['p1']
            slot_prompt, agent_response = agents[current_agent]['instance'].execute_round(history['content'], round_idx)
            if not args.dry_run_on_history:
                history = save_conversation(history, current_agent, agent_response, slot_prompt, agents=agents, round_assign=agent_round_assignment, initial=True)
            else:
                n_input_tokens.append(agents[current_agent]['instance'].n_input_tokens[-1])
                n_output_tokens.append(agents[current_agent]['instance'].n_output_tokens[-1])
            print('=====')
            print(f'{current_agent} response: {agent_response}')
        #Continue with rounds 
        current_agent = agent_round_assignment[round_idx]
        slot_prompt, agent_response = agents[current_agent]['instance'].execute_round(history['content'], round_idx)
        if not args.dry_run_on_history:
            history = save_conversation(history, current_agent, agent_response, slot_prompt)
        else:
            n_input_tokens.append(agents[current_agent]['instance'].n_input_tokens[-1])
            n_output_tokens.append(agents[current_agent]['instance'].n_output_tokens[-1])
    print('=====')
    print(f'{current_agent} response: {agent_response}')

#Final deal by P1 
print(" ==== Deal Suggestions ==== ")
current_agent = role_to_agent_names['p1']
slot_prompt, agent_response = agents[current_agent]['instance'].execute_round(history['content'], args.rounds_num)
if not args.dry_run_on_history:
    history = save_conversation(history, current_agent, agent_response, slot_prompt)
else:
    n_input_tokens.append(agents[current_agent]['instance'].n_input_tokens[-1])
    n_output_tokens.append(agents[current_agent]['instance'].n_output_tokens[-1])
print('=====')
print(f'{current_agent} response: {agent_response}')

if args.dry_run_on_history:
    print('max input tokens:', max(n_input_tokens))
    print('max output tokens:', max(n_output_tokens))
    print('max input + output tokens:', max(map(sum, zip(n_input_tokens, n_output_tokens))))
    print('total input + output tokens:', sum(map(sum, zip(n_input_tokens, n_output_tokens))))

if args.api_key and not args.azure:
    print('\n====================================================\n')
    input_tokens, output_tokens, cost = agents[current_agent]['instance'].return_tokens()
    print(f"Tokens --- Inputs: {input_tokens}, Outputs: {output_tokens}")
    print(f"Run cost: ${cost}")
    print('\n====================================================\n')