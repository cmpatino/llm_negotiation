import os 
import re
import string
import json

import numpy as np


def load_setup(output_dir, num_agents, num_issues):
    
    with open(os.path.join(output_dir, 'config.txt'), 'r') as f:
        agents_config_file = f.readlines()

    issue_names = string.ascii_uppercase[:26]

    agents = {}
    role_to_agents = {}
    incentive_to_agents = {}
    
    assert len(agents_config_file) == num_agents
    
    for line in agents_config_file:   
        agent_game_name, file_name, role, incentive, model = line.split(',') 
        model = model.strip()
        agents[agent_game_name] = {'file_name': file_name, 'role': role, 'incentive': incentive }
        if not role in role_to_agents: role_to_agents[role] = []
        if not incentive in  incentive_to_agents: incentive_to_agents[incentive] = []
        role_to_agents[role].append(agent_game_name)
        incentive_to_agents[incentive].append(agent_game_name)
    
    for agent in agents: 
        scores = {}
        with open(os.path.join(output_dir, 'scores_files', agents[agent]['file_name']) + '.txt', 'r') as f: 
            Lines = f.readlines()
            assert len(Lines) == num_issues + 1
            for i, line in enumerate(Lines):
                if i == len(Lines) - 1: #min thresholds 
                    scores['min'] = int(line.strip())
                    break
                scores[issue_names[i]] = [int(num.strip()) for num in line.split(',')]
        agents[agent]['scores'] = scores 
        
    for role in role_to_agents:
        if len(role_to_agents[role]) == 1: 
            role_to_agents[role] = role_to_agents[role][0]
            
    for incentive in incentive_to_agents:
        if len(incentive_to_agents[incentive]) == 1: 
            incentive_to_agents[incentive] = incentive_to_agents[incentive][0]
            
    return agents, role_to_agents, incentive_to_agents


def calculator(scores, deal, num_issues=5):
    if len(deal) != num_issues: return 0 
    deal_sum = 0
    for issue in deal:
        if issue == '' or len(issue)!= 2 : return 0
        issue, number = issue[0], int(issue[1])
        if issue not in scores: return 0 
        deal_sum += scores[issue][number-1]
    return deal_sum 


def extract_deal(answer, num_issues=5): 
    answer = answer.replace('\n','')
    issue_names = string.ascii_uppercase[:26] 
    deal = []
    issues_suggested=0
    for i in range(0,num_issues):
        option = re.findall(f'{issue_names[i]}[1-9]', answer,re.DOTALL)
        deal.append(option[0]) if option else deal.append('')
        if option: issues_suggested +=1 
    return deal, issues_suggested


def loop_all_parties(agents, deal, veto_parties, num_issues):
    '''
    loop over all parties and calculate sum of scores for a particular deal made by any agent at any point 
    calculate number of agreeing parties
    return whether veto parties agreed 
    
    Returns:
        - number of agreeing parties 
        - whether veto parties agree 
        - a list of scores of parties for that deal 
    '''
    agreed = 0 
    veto_agreed = [False for _ in veto_parties]
    all_parties_score = []

    for name_i in agents.keys():
        party_score = calculator(agents[name_i]['scores'], deal, num_issues) 
        all_parties_score.append(party_score)
        if party_score > agents[name_i]['scores']['min']: 
            agreed += 1 
            if name_i in veto_parties: veto_agreed[veto_parties.index(name_i)] = True 
                
    return agreed, veto_agreed, all_parties_score

def check_correctness(agents, agent_name, p1, deal, veto_agreed, wrong_suggested, num_issues):
    '''
    check if a deal is valid wrt the min score of the agent 
    if the agent is p1 and (deal_value+10) > threshold, consider it valid and correct veto_agreed 
    returns:
        updated value of wrong suggested 
        updated veto_agreed 
    '''
    
    deal_value =  calculator(agents[agent_name]['scores'],deal, num_issues) 
    
    if deal_value < agents[agent_name]['scores']['min']: 
        if agent_name == p1 and (deal_value + 10) >= agents[agent_name]['scores']['min']:
            veto_agreed[0] = True
        else:
            # not p1 
            # p1 and not (deal_value+10) > thresholds (to accomodate the bonus rule)
            wrong_suggested += 1 
    return wrong_suggested, veto_agreed


def check_agreement(agents, p1, agreed, veto_agreed, deal, num_issues, num_agents):
    '''
    check if current deal leads to agreement (by a deal made by p1)
    
    if all party agrees and (deal_value+10) >= min score of p1 -> all agreement, deal is done (ANY metric)
    
    if 4 party agree including p2 and >= min -> 5 way agreement, deal is done (ANY metric)
    
    
    Returns:
        - whether a deal can be done based on this current round 
        - whether there is all_agreement
    '''
    
    deal_value =  calculator(agents[p1]['scores'], deal, num_issues) 

    curr_round_deal_done = False 
    all_agreement = False 
    
    if agreed == num_agents: 
        curr_round_deal_done = True 
        all_agreement = True        
    
    elif agreed == (num_agents-1) and (not veto_agreed[0]) and ( (deal_value+10) >= agents[p1]['scores']['min'] ):
        #p1 not met but would be met with +10 rule  
        print(deal)
        curr_round_deal_done = True 
        all_agreement = True
        
    elif agreed == (num_agents-1) and all(veto_agreed): 
        #one other party was excluded. 
        curr_round_deal_done = True 

        
    return curr_round_deal_done,all_agreement



def get_metrics(agents, answers, veto_parties, num_rounds, num_issues, num_agents):
    
    '''
    compute metrics for one negotiation session
    '''

    deal_values_per_agents = {name_:[] for name_ in agents.keys()} 
    wrong_suggested = 0 
    all_deals_count = 0 
    deal_done = False
    
    p1 = veto_parties[0]
    
    # rounds should be number of rounds + 2 (start and final deal suggestion)
    if len(answers['rounds']) != num_rounds+2: return 
    
    for round_ in answers['rounds']:
        name,answer = round_['agent'], round_['public_answer']
        deal, issues_suggested = extract_deal(answer,num_issues) 

        if issues_suggested < num_issues: continue
            
        veto_agreed = [False for _ in range(len(veto_parties))]
        
        all_deals_count += 1 
        
        agreed, veto_agreed, all_parties_score = loop_all_parties(agents, deal, veto_parties, num_issues)

        wrong_suggested, veto_agreed = check_correctness(agents,name, p1,deal,veto_agreed,wrong_suggested, num_issues)
        
        if name == p1: 
            curr_deal_done, all_agreement = check_agreement(agents,p1,agreed,veto_agreed,deal, num_issues, num_agents)
        
        deal_done = curr_deal_done or deal_done
        
        deal_value = calculator(agents[name]['scores'],deal,num_issues)
        deal_values_per_agents[name].append([deal_value,agreed,all_parties_score])
    
    print('Final deal: ' + ', '.join(deal))
    
    '''
    returns 
        - wrong_suggested/all_deals_count (ratio of wrong suggested deals)
        - deal_done (whether a deal has been done in any round by p1)
        - all_agreement (whether the final deal by p1 led to all agreement)
        - curr_deal_done (whether the final deal by p1 led to an agreement)
        - deal_values_self: a dict of agents and their suggested deals: value to themselves, number of agreeing, average score of parties
    '''
    return wrong_suggested / all_deals_count, deal_done, all_agreement, curr_deal_done, deal_values_per_agents


def get_curves(output_dir, agents_num, issues_num, num_rounds, agent_name, type):
    agents, role_to_agents, incentive_to_agents = load_setup(output_dir, agents_num, issues_num)
    answers_files = [os.path.join(output_dir, filename) for filename in os.listdir(output_dir) if filename.startswith("history")]

    wrong_suggested, deal_done, all_agreement, deal_final_round, deal_values_per_agents = [], [], [], [], []
    for file_ in answers_files:
        # print(file_)

        answers = json.load(open(file_))
        metrics =  get_metrics(agents, answers, [role_to_agents['p1'], role_to_agents['p2']], num_rounds, issues_num, agents_num)

        if metrics: 
            wrong_suggested.append(metrics[0])
            deal_done.append(metrics[1])
            all_agreement.append(metrics[2])
            deal_final_round.append(metrics[3])
            deal_values_per_agents.append(metrics[4])
        else:
            print('Error in file: ' + file_)

    suggested = {'own_value':[], 'agree':[], 'others_scores' : [], 'avg_value': []}
    for session_i, session in enumerate(deal_values_per_agents):
        session_own = []
        session_agree = []
        session_others_scores = []
        session_avg_scores = []

        for i, round_ in enumerate(session[agent_name]): 
            session_own.append(round_[0])
            session_agree.append(round_[1])
            session_others_scores.append(round_[-1])
            session_avg_scores.append(np.mean(round_[-1]))
            
        suggested['own_value'].append(session_own)
        suggested['agree'].append(session_agree)
        suggested['others_scores'].append(session_others_scores)
        suggested['avg_value'].append(session_avg_scores)

    # Compute average of metrics of the same round (e.g., round 0), across all negotiation sessions 

    ## own score 
    average_round_own = []
    std_round_own = []

    if type == 'multi':
        range_ = int(num_rounds/agents_num)+2
    elif type == 'single':
        range_ = 6

    for j in range(0, range_):
        average_round_own.append(np.mean([suggested['own_value'][i][j] for i in range(len(suggested['own_value']))]))
        std_round_own.append(np.std( [suggested['own_value'][i][j] for i in range(len(suggested['own_value']))]))


    ## collective score 
    average_collective_deals = []
    std_collective_deals = []

    for j in range(0, range_):
        average_collective_deals.append(np.mean( [suggested['avg_value'][i][j] for i in range(len(suggested['avg_value']))]))
        std_collective_deals.append(np.std( [suggested['avg_value'][i][j] for i in range(len(suggested['avg_value']))]))
    return average_round_own, std_round_own, average_collective_deals, std_collective_deals

def gini_coefficient(values):
    """
    Parameters:
    values (list or numpy array): A list or array of numerical values (e.g., gains of agents).
    Returns:
    float: Gini coefficient, ranging from 0 (perfect equality) to 1 (maximum inequality).
    """
    if not values or len(values) == 0:
        raise ValueError("The list of values must not be empty.")

    values = np.array(values, dtype=np.float64)
    n = len(values)
    mean_value = np.mean(values)

    if mean_value == 0:
        return 0.0  #avoid division by zero

    #sum of absolute differences
    absolute_differences_sum = sum(abs(values[i] - values[j]) for i in range(n) for j in range(n))
    #gini coefficient formula
    gini = absolute_differences_sum / (2 * n**2 * mean_value)

    return gini

def check_answers_in_file(filepath):
    with open(filepath, 'r') as f:
        history = json.load(f)
    #extract rounds
    rounds = history.get('rounds', [])
    print(f"File: {filepath}")
    mismatches = []
    for i, turn in enumerate(rounds):
        full_answer = turn.get('full_answer', '')
        public_answer = turn.get('public_answer', '')
        if (full_answer == public_answer or any(tag in public_answer for tag in ["<PLAN>", "</PLAN>", "<SCRATCHPAD>", "</SCRATCHPAD>"])):
            mismatches.append({
                "round": i + 1,
                "agent": turn.get('agent', 'unknown'),
                "full_answer": full_answer,
                "public_answer": public_answer
            })
    return mismatches