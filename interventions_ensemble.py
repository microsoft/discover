

import os
os.environ['HF_HOME'] = '.transformers/'

import logging
import math
import json
import copy

import torch
from torch import optim

from transformers import AutoTokenizer
from transformers import GptOssForCausalLM
from transformers import DynamicCache


from utils import *
from models import *
from evaluation import *
from training import *

import argparse
from discover_argparser import discover_parser



# Format of the intervention files:
# 1. In the simplest case, the interventions are just changing a filler. This will
#    be indicated with pairs like fr_pairs_old=(DUMMY,artist), fr_pairs_new=(DUMMY,sailor).
#    This means: In any filler role pair in which "artist" is the filler, change it to 
#    "sailor" (DUMMY stands for the role, meaning that we ignore what the role is).
#    These use the command-line argument --intervention_all_bidirectional (or, if we're doing
#    a constrained version, --intervention_other_bidirectional --intervention_no_check_tpr or
#    --intervention_self_bidirectional --intervention_no_check_tpr)
# 2. In other cases, a role-filler pair is being changed. This will be indicated with
#    pairs like fr_pairs_old=(subj_adj,famous), fr_pairs_new=(subj_rc_subj_pp_adj,famous). All
#    changes will be made required to change one filler-role pair to another (since the roles are
#    assumed to be bidirectional, this is complex because the roles to be replaced, such as "subj_adj",
#    are unidirectional, and we need to revise any bidirectional role that contains this role as one
#    of its parts. These use the command-line argument --intervention_all_bidirectionalspec (or, if we're
#    doing a constrained version, --intervention_other_bidirectionalspec --intervention_no_check_tpr or
#    --intervention_self_bidirectionalspec --intervention_no_check_tpr
# 3. Sometimes multiple filler-role pairs are to be changed, in which case they are separated by semicolons
#    E.g., fr_pairs_old=(obj_pp_p,next);(obj_pp_p,to);(obj_pp_det,the);(obj_pp_pobj,reporter)
#          fr_pairs_new=(subj_pp_p,next);(subj_pp_p,to);(subj_pp_det,the);(subj_pp_pobj,reporter)





# Load arguments
parser = argparse.ArgumentParser(parents=[discover_parser], description="Arguments governing model architectures and training")

args = parser.parse_args()

random_seed = args.random_seed
log_suffix = args.log_suffix

if args.json_args is not None:
    with open(args.json_args, 'rt') as f:
        copy_args = argparse.Namespace()
        copy_args.__dict__.update(json.load(f))
        args = parser.parse_args(namespace=copy_args)

# So that the random seed and log suffix are not restricted to be
# the same as in the loaded args
args.random_seed = random_seed
args.log_suffix = log_suffix

random.seed(args.random_seed)
np.random.seed(args.random_seed)
torch.manual_seed(args.random_seed)


####################################################################
# Preliminaries
####################################################################

# Check if we are on a GPU
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


# Create filenames for saving logs and weights
savename = "tpe_ensemble_" + args.llm_name + "_" + args.discover_role_scheme + args.log_suffix
if args.intervention_foil is not None:
    log_name = args.log_dir + savename  + "_replace_" + args.intervention_type + "-" + args.intervention_foil + ".log"
else:
    log_name = args.log_dir + savename  + "_replace_" + args.intervention_type + ".log"
if args.intervention_all_bidirectionalspec or args.intervention_all_bidirectional:
    log_name = log_name.replace("replace", "replaceall")
elif args.intervention_self_bidirectionalspec or args.intervention_self_bidirectional:
    log_name = log_name.replace("replace", "replaceself")
elif args.intervention_other_bidirectionalspec or args.intervention_other_bidirectional:
    log_name = log_name.replace("replace", "replaceother")
save_model_name = args.weight_dir + savename + ".weights"

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, force=True, handlers=[logging.StreamHandler(),logging.FileHandler(log_name)])




####################################################################
# Set up the target LLM to be decomposed
####################################################################

if args.llm_name == "gpt-oss":
    llm_model_name = "openai/gpt-oss-20b"
    tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
    llm_model = GptOssForCausalLM.from_pretrained(llm_model_name, torch_dtype=torch.bfloat16, device_map="auto")
llm_model.eval()
llm_model.requires_grad_(False) 


if args.prefix_filename is not None:
    fi = open(args.data_dir + args.prefix_filename, "r")
    prefix = fi.readline().strip().replace("\\n", "\n").replace("\\t", "\t")
    
    args.prefix_str_length = len(prefix.split())

    tokens = tokenizer([prefix], return_tensors="pt")
    args.prefix_length = len(tokens["input_ids"][0])

logging.info(args)

args.truncate_token = None
if args.truncate_after_strs is not None:
    tokens = tokenizer([args.truncate_after_strs.split(",")[0]], return_tensors="pt")["input_ids"][0]
    if len(tokens) == 1:
        args.truncate_token = tokens[0].item()


####################################################################
# Create data loaders
####################################################################

prefix = args.data_dir + "sentences"

train_set = EnsembleDataLoader(prefix + ".train", args.batch_size, stream=True, batches_per_buffer=1, shuffle=True, headers=True, encoding_model=llm_model, tokenizer=tokenizer, add_space=args.add_space, prefix_length=args.prefix_length, suffix_length=args.suffix_length)
valid_set = EnsembleDataLoader(prefix + ".valid", args.batch_size, stream=True, batches_per_buffer=1, shuffle=True, headers=True, encoding_model=llm_model, tokenizer=tokenizer, add_space=args.add_space, prefix_length=args.prefix_length, suffix_length=args.suffix_length)
test_set = EnsembleDataLoader(prefix + ".test", 1, stream=True, batches_per_buffer=1, shuffle=False, headers=True, encoding_model=llm_model, tokenizer=tokenizer, discover_batch_size=1000, add_space=args.add_space, prefix_length=args.prefix_length, suffix_length=args.suffix_length)

intervention_set = EnsembleDataLoader(args.data_dir + "interventions." + args.intervention_type, 1, stream=True, batches_per_buffer=1, shuffle=False, headers=True, encoding_model=llm_model, tokenizer=tokenizer, discover_batch_size=1000, add_space=args.add_space, prefix_length=args.prefix_length, suffix_length=args.suffix_length)
intervention_eval_set = EnsembleDataLoader(args.data_dir + "interventions_eval." + args.intervention_type, 1, stream=True, batches_per_buffer=1, shuffle=False, headers=True, encoding_model=llm_model, tokenizer=tokenizer, discover_batch_size=1000, add_space=args.add_space, prefix_length=args.prefix_length, suffix_length=args.suffix_length)

if args.intervention_foil is not None:
    intervention_foil_set = EnsembleDataLoader(args.data_dir + "interventions." + args.intervention_foil, 1, stream=True, batches_per_buffer=1, shuffle=False, headers=True, encoding_model=llm_model, tokenizer=tokenizer, discover_batch_size=1000, add_space=args.add_space, prefix_length=args.prefix_length, suffix_length=args.suffix_length)
    intervention_foil_eval_set = EnsembleDataLoader(args.data_dir + "interventions_eval." + args.intervention_foil, 1, stream=True, batches_per_buffer=1, shuffle=False, headers=True, encoding_model=llm_model, tokenizer=tokenizer, discover_batch_size=1000, add_space=args.add_space, prefix_length=args.prefix_length, suffix_length=args.suffix_length)
else:
    intervention_foil_set = [None] * 1000
    intervention_foil_eval_set = [None] * 1000



####################################################################
# Create ensemble of tensor product encoders
####################################################################

seq2roles = RoleAssigner(args.discover_role_scheme, filler_indices=True, tokenizer=tokenizer, memoize=False, prefix_length=args.prefix_length, suffix_length=args.suffix_length) 


logging.info("Updating roles")

filler_role_save_prefix = args.data_dir + args.discover_role_scheme
if args.discover_role_scheme + ".filler_token2idx.json" in os.listdir(args.data_dir):
    seq2roles.update_from_files(filler_role_save_prefix)
else:
    seq2roles.update_for_ensemble_dataset(train_set)
    logging.info("Updated for train")
    seq2roles.update_for_ensemble_dataset(valid_set)
    logging.info("Updated for valid")
    seq2roles.update_for_ensemble_dataset(test_set)
    logging.info("Updated for test")

    seq2roles.save_to_files(filler_role_save_prefix)


n_fillers = seq2roles.filler_vocab_size
n_roles = seq2roles.role_vocab_size


logging.info("Number of fillers: " + str(n_fillers))
logging.info("Number of roles: " + str(n_roles))
tpe_ensemble = TensorProductEncoderEnsemble(ensemble_size=llm_model.config.num_hidden_layers+1, hidden_size=llm_model.config.hidden_size, n_fillers=n_fillers, n_roles=n_roles, filler_dim=args.discover_emb_size, role_dim=args.discover_role_emb_size, seq2roles=seq2roles, has_linear_layer=True).to(device=device)

logging.info("Made tpe_ensemble")



logging.info("computing standardization")
mean_vector_list, stddev_vector_list = compute_standardization(train_set, n_layers=llm_model.config.num_hidden_layers+1)
train_set.mean_vector_list = mean_vector_list
train_set.stddev_vector_list = stddev_vector_list
valid_set.mean_vector_list = mean_vector_list
valid_set.stddev_vector_list = stddev_vector_list
test_set.mean_vector_list = mean_vector_list
test_set.stddev_vector_list = stddev_vector_list
intervention_set.mean_vector_list = mean_vector_list
intervention_set.stddev_vector_list = stddev_vector_list
if args.intervention_foil is not None:
    intervention_foil_set.mean_vector_list = mean_vector_list
    intervention_foil_set.stddev_vector_list = stddev_vector_list

tpe_ensemble.load_state_dict(torch.load(save_model_name))
tpe_ensemble.eval()


delete_strings = []
for char in args.delete_strings:
    delete_strings.append(char)


####################################################################
# Manipulate encodings
####################################################################

def halve_role(role):
    if role.startswith("0_"):
        parts = role[2:]
    else:
        parts = role
    parts = parts.split("-")
    first_half = parts[0]
    return first_half


def split_role(role):
    if role.startswith("0_"):
        role = role[2:]

    parts = role.split("-")
    if len(parts) != 2:
        logging.info("Invalid role: " + role)
        15/0

    return parts[0], parts[1]

def remove_parens(fr_string):
    if fr_string[0] == "(":
        return remove_parens(fr_string[1:])
    elif fr_string[-1] == ")":
        return remove_parens(fr_string[:-1])
    else:
        return fr_string

def split_on_comma(fr_string):
    ind_comma = fr_string.index(",")
    return [fr_string[:ind_comma], fr_string[ind_comma+1:]]


# tpe_3 was chosen arbitrarily - any TPE would work, since
# they all use the same role scheme
all_roles = tpe_ensemble.tpe_3.seq2roles.role_token2idx
atomic_roles = {}
for role in all_roles:
    halved_role = halve_role(role)
    if role == "0_" + halved_role + "-" + halved_role:
        atomic_roles[halved_role] = 1




total_interventions = 0                 # Total number of interventions in dataset
total_interventions_correct = 0         # Total number of intervention examples with the correct answer when starting with the LLM encoding and adding/subtracting targeted TPE pairs
total_tpe_interventions_correct = 0     # Total number of intervention examples with the correct answer when starting with the TPE encoding and adding/subtracting targeted TPE pairs 
total_llm_correct = 0                   # Total number of examples where the LLM gives the right answer on the original, unmodified prompt
total_llm_correct_eval = 0              # Total number of examples where the LLM gives the right answer on the original, unmodified prompt that should result from the intervention
total_tpe_correct = 0                   # Total number of examples where the LLM gives the right answer on the original, unmodified prompt when fed the TPE approximation of its hidden states
total_tpe_agreement = 0                 # Total number of examples where the LLM gives the same answer on the original, unmodified prompt when fed the TPE approximation of its hidden states or its actual hidden states
total_tpe_correct_eval = 0              # Total number of examples where the LLM gives the right answer when fed the TPE approximation of its hidden states for the prompt that should result from the intervention
total_tpe_agreement_eval = 0            # Total number of examples where the LLM gives the same answer when fed its own hidden states or the TPE approximation of its hidden states for the prompt that should result from the intervention
total_interventions_agreement = 0       # Total number of examples where the LLM gives the same answer when fed its own hidden states or the result of doing a TPE intervention on its hidden states
total_tpe_interventions_agreement = 0   # Total number of examples where the LLM gives the same answer when fed a TPE approximation of its own hidden states or when fed its own hidden states for the eval
total_replace_correct = 0               # Total number of times when replacing the LLM hidden states with the "add" TPE pairs results in the correct answer
total_replace_agreement = 0             # Total number of times when replacing the LLM hidden states with the "add" TPE pairs results in the same answer that the LLM itself gets on the prompt that should result from interventions

if args.intervention_foil is not None:
    total_llm_correct_foil = 0              # Total number of examples where the LLM gives the right answer on the original, unmodified foil prompt
    total_llm_correct_eval_foil = 0         # Total number of examples where the LLM gives the right answer on the prompt that should result from intervening on the foil prompt
    total_interventions_correct_foil = 0    # Total number of examples where the LLM gives the right answer when starting with the foil prompt and then adding/subtracting targeted TPE pairs
    total_interventions_agreement_foil = 0 # Total number of examples where the LLM gives the same answer when starting with the foil prompt and then adding/subtracting targeted TPE pairs, vs when using its own hidden states for the prompt that results from the intervention


def get_fr_string(orig_f, orig_r, to_add_f, to_add_r, to_subtract_f, to_subtract_r):
    orig_pairs = list(zip(orig_f, orig_r))
    to_add_pairs = list(zip(to_add_f, to_add_r))
    to_subtract_pairs = list(zip(to_subtract_f, to_subtract_r))

    orig_pairs = [elt[0] + "&&&" + elt[1] for elt in orig_pairs if "<PAD>" not in elt]
    to_add_pairs = [elt[0] + "&&&" + elt[1] for elt in to_add_pairs if "<PAD>" not in elt]
    to_subtract_pairs = [elt[0] + "&&&" + elt[1] for elt in to_subtract_pairs if "<PAD>" not in elt]
    all_frs = orig_pairs + to_add_pairs
    updated = []

    for candidate_fr in all_frs:
        if candidate_fr not in to_subtract_pairs:
            updated.append(candidate_fr)
        else:
            if all_frs.count(candidate_fr) == 2 and to_subtract_pairs.count(candidate_fr) == 1 and candidate_fr not in updated:
                updated.append(candidate_fr)

    updated = sorted(updated)

    updated = "%%%".join(updated)
    return updated




index = 0
for index, (examples, examples_eval, examples_foil, examples_foil_eval) in enumerate(zip(intervention_set, intervention_eval_set, intervention_foil_set, intervention_foil_eval_set)):

    indices_to_replace = {}

    output_list, hidden_list = tpe_ensemble(examples)
    
    fillers, role_lists = tpe_ensemble.tpe_3.seq2roles.assign_roles_batch(examples[11])
    all_fillers = fillers[:]
    all_role_lists = role_lists[:]

    fillers_eval, role_lists_eval = tpe_ensemble.tpe_3.seq2roles.assign_roles_batch(examples_eval[11])


    # Headers in each example: new_answer_str  new_answer      fr_pairs_old    fr_pairs_new
    fr_pairs_old = examples[0]["fr_pairs_old"][0].split(";")
    fr_pairs_new = examples[0]["fr_pairs_new"][0].split(";")
    fr_replacements = list(zip(fr_pairs_old, fr_pairs_new))
    fr_replacements_processed = []
    max_n_replacements = -1
    for old, new in fr_replacements:
        
        old_fr_pairs = [remove_parens(old)]
        old_fr_pairs_tuples = [split_on_comma(elt) for elt in old_fr_pairs]
        
        new_fr_pairs = [remove_parens(new)]
        new_fr_pairs_tuples = [split_on_comma(elt) for elt in new_fr_pairs]

        fr_replacements_processed.append((old_fr_pairs_tuples, new_fr_pairs_tuples))

        n_replacements = len(old_fr_pairs)
        if n_replacements > max_n_replacements:
            max_n_replacements = n_replacements+50

        
    # hidden_list[0].shape[1] is the batch size
    fillers_to_subtract = [[0]*max_n_replacements for _ in range(hidden_list[0].shape[1])]
    roles_to_subtract = [[0]*max_n_replacements for _ in range(hidden_list[0].shape[1])]

    fillers_to_add = [[0]*max_n_replacements for _ in range(hidden_list[0].shape[1])]
    roles_to_add = [[0]*max_n_replacements for _ in range(hidden_list[0].shape[1])]

    # For debugging purposes: Here, we construct the full set of filler/role pairs
    # that should be present. Then we can check if the filler/role pairs that we
    # are swapping in and out should change the original prompt to the target one
    target_fr_combos = {}
    for fr_index, (filler_eval_list, role_eval_list) in enumerate(zip(fillers_eval, role_lists_eval[0])):
        fillers_eval = [tpe_ensemble.tpe_3.seq2roles.filler_idx2token[str(elt)] for elt in filler_eval_list]
        roles_eval = [tpe_ensemble.tpe_3.seq2roles.role_idx2token[str(elt)] for elt in role_eval_list]

        all_pairs = []
        for f_eval, r_eval in zip(fillers_eval, roles_eval):
            combo = str(f_eval) + "&&&" + str(r_eval)
            if f_eval != "<PAD>":
                all_pairs.append(combo)
        all_pairs = sorted(all_pairs)
        full_fr_list_target = "%%%".join(all_pairs)
        target_fr_combos[full_fr_list_target] = 1

   
    # We iterate over each token and figure out whether the fillers and roles present 
    # in that token necessitate modifying anything
    for fr_index, (filler_list, role_list) in enumerate(zip(fillers, role_lists[0])):
        fillers = [tpe_ensemble.tpe_3.seq2roles.filler_idx2token[str(elt)] for elt in filler_list]
        roles = [tpe_ensemble.tpe_3.seq2roles.role_idx2token[str(elt)] for elt in role_list]

        pairs = list(zip(roles, fillers))
        pairs_no_pad = []
        for pair in pairs:
            if pair[0] != "<PAD>":
                pairs_no_pad.append(pair)

        padded_add_fillers = []
        padded_add_roles = []
        padded_subtract_fillers = []
        padded_subtract_roles = []


        if args.intervention_all_bidirectional:
            # Assume there's only one basic filler/role pair being replaced
            # Replace all filler/role pairs that contain the filler to be
            #   replaced, at all positions
            fr_replacement = fr_replacements_processed[0]
            r_to_replace, f_to_replace = fr_replacement[0][0]
            r_to_use, f_to_use = fr_replacement[1][0]
            for pair_r, pair_f in pairs_no_pad:
                # Don't bother with roles, just assume filler uniqueness
                if pair_f == f_to_replace:
                    padded_subtract_fillers += [f_to_replace]
                    padded_subtract_roles += [pair_r]
                    padded_add_fillers += [f_to_use]
                    padded_add_roles += [pair_r]

        elif args.intervention_self_bidirectional:
            # Assume there's only one basic one being replaced
            # Only replace at the indices where the filler in question is the token
            fr_replacement = fr_replacements_processed[0]
            r_to_replace, f_to_replace = fr_replacement[0][0]
            r_to_use, f_to_use = fr_replacement[1][0]
            for pair_r, pair_f in pairs_no_pad[-1:]:
                # Don't bother with roles, just assume filler uniqueness
                if pair_f == f_to_replace:
                    padded_subtract_fillers += [f_to_replace]
                    padded_subtract_roles += [pair_r]
                    padded_add_fillers += [f_to_use]
                    padded_add_roles += [pair_r]

                    indices_to_replace[fr_index] = 1


        elif args.intervention_other_bidirectional:
            # Assume there's only one basic one being replaced
            # Only replace at the indices where the filler in question is the token
            # (i.e., the opposite of intervention_self_bidirectional)
            fr_replacement = fr_replacements_processed[0]
            r_to_replace, f_to_replace = fr_replacement[0][0]
            r_to_use, f_to_use = fr_replacement[1][0]
            for pair_r, pair_f in pairs_no_pad[:-1]:
                # Don't bother with roles, just assume filler uniqueness
                if pair_f == f_to_replace:
                    padded_subtract_fillers += [f_to_replace]
                    padded_subtract_roles += [pair_r]
                    padded_add_fillers += [f_to_use]
                    padded_add_roles += [pair_r]
        
        elif args.intervention_all_bidirectionalspec or args.intervention_self_bidirectionalspec or args.intervention_other_bidirectionalspec: 
            
            all_pairs_to_replace = [[fr_replacement[0][0][0], fr_replacement[1][0][0]] for fr_replacement in fr_replacements_processed]
            all_roles_to_replace = {}
            for elt1, elt2 in all_pairs_to_replace:
                all_roles_to_replace[elt1] = elt2


            for fr_replacement in fr_replacements_processed:
                  
                # Tells us what the role of the curent token is
                current_token_role = halve_role(pairs_no_pad[-1][0])
                replacement_current_token_role = None
                replacement_current_token_filler = None

                replace_roles = [elt[0][0][0] for elt in fr_replacements_processed]
                
                if current_token_role in replace_roles:
                    current_token_role_index = replace_roles.index(current_token_role)
                    replacement_current_token_role = fr_replacements_processed[current_token_role_index][1][0][0]
                    replacement_current_token_filler = fr_replacements_processed[current_token_role_index][1][0][1]

                # Assume there's only one basic one being replaced
                r_to_replace, f_to_replace = fr_replacement[0][0]
                r_to_use, f_to_use = fr_replacement[1][0]


                # If the current token's role is one to replace
                if current_token_role == r_to_replace:

                    # Iterate over the pairs in the current token
                    for pair_r, pair_f in pairs_no_pad:
                        
                        pair_r_r1, pair_r_r2 = split_role(pair_r)
                        
                        # Create a new role by replacing each sub-role
                        # that has a replacement
                        if pair_r_r1 in all_roles_to_replace:
                            new_r1 = all_roles_to_replace[pair_r_r1]
                        else:
                            new_r1 = pair_r_r1
                        
                        if pair_r_r2 in all_roles_to_replace:
                            new_r2 = all_roles_to_replace[pair_r_r2]
                        else:
                            new_r2 = pair_r_r2
                        candidate_role = "0_" + new_r1 + "-" + new_r2


                        if not args.intervention_other_bidirectionalspec:
                            # These are strictly self roles

                            indices_to_replace[fr_index] = 1

                            # Check if the candidate replacement is a possible role. If so, subtract
                            # the existing one and add in the replacement. Else, just subtract
                            # out the existing one.
                            if candidate_role in tpe_ensemble.tpe_3.seq2roles.role_token2idx: # Means that the candidate role is a valid role
                                
                                # Check if this candidate role has already been added in to this position
                                replaced_already = False
                                if pair_r == "0_" + r_to_replace + "-" + r_to_replace and pair_f == f_to_replace:
                                    potential_f = pair_f.replace(f_to_replace, f_to_use)
                                else:
                                    potential_f = pair_f
                                if (candidate_role, pair_f) in list(zip(padded_add_roles, padded_add_fillers)):
                                    replaced_already = True

                                # If not already added: Add it in, and subtract the one to be removed
                                if not replaced_already:
                                    padded_add_roles += [candidate_role]
                                    padded_subtract_roles += [pair_r]
                                   
                                    # Change the filler if this is the role-filler pair to be replaced; else keep the filler
                                    if pair_r == "0_" + r_to_replace + "-" + r_to_replace and pair_f == f_to_replace:
                                        padded_add_fillers += [pair_f.replace(f_to_replace, f_to_use)]
                                    else:
                                        padded_add_fillers += [pair_f]
                                    padded_subtract_fillers += [pair_f]


                            else: # Means that the candidate role is not a valid role, so we subtract it but don't add a new one
                                if (pair_r, pair_f) not in list(zip(padded_subtract_roles, padded_subtract_fillers)):
                                    padded_subtract_roles += [pair_r]
                                    padded_subtract_fillers += [pair_f]

                   

                    # Now figure out if there are any we need to add

                    # All the fillers and roles that are present in the entire sequence
                    complete_fillers = all_fillers[-1][:]
                    complete_roles = all_role_lists[0][-1][:]
                    complete_fillers = [elt for elt in complete_fillers if elt != "<PAD>"]
                    complete_roles = [elt for elt in complete_roles if elt != "<PAD>"]
                    complete_fillers = [tpe_ensemble.tpe_3.seq2roles.filler_idx2token[str(elt)] for elt in complete_fillers]
                    complete_roles = [tpe_ensemble.tpe_3.seq2roles.role_idx2token[str(elt)] for elt in complete_roles]

                    # Consider each r-f pair in the whole sequence
                    for this_r, this_f in zip(complete_roles, complete_fillers):
                        
                        # Create the candidate role that this r-f pair would
                        # form inside the current token
                        sentence_position, _ = split_role(this_r)
                        this_f_replaced = this_f[:]


                        for inner_fr_replacement in fr_replacements_processed:
                            inner_r_to_replace, inner_f_to_replace = inner_fr_replacement[0][0]
                            inner_r_to_use, inner_f_to_use = inner_fr_replacement[1][0]
                            if inner_r_to_replace == sentence_position and inner_f_to_replace == this_f:
                                this_f_replaced = inner_f_to_use
                                sentence_position = inner_r_to_use

                        candidate_role = "0_" + sentence_position + "-" + r_to_use


                        # It's a valid role and hasn't been used yet
                        if candidate_role in tpe_ensemble.tpe_3.seq2roles.role_token2idx and (candidate_role, this_f_replaced) not in list(zip(padded_add_roles, padded_add_fillers)):
                           
                            # This checking is necessary for cases where a role appears
                            # multiple times (e.g., when there is a multi-word preposition)
                            if candidate_role == "0_" + r_to_use + "-" + r_to_use:

                                # The number of "self" roles added already
                                count_added = 0
                                for role_added in padded_add_roles:
                                    if role_added == candidate_role:
                                        count_added += 1

                                # The number of "self" roles present in this token
                                count_to_replace = 0
                                for role in roles:
                                    if role == "0_" + r_to_replace + "-" + r_to_replace:
                                        count_to_replace += 1

                                # If we've already added as many self roles as we need,
                                # don't add any more
                                if count_added >= count_to_replace:
                                    continue

                            replaced_already = False
                            if (candidate_role, this_f_replaced) in list(zip(padded_add_roles, padded_add_fillers)):
                                replaced_already = True

                            if not replaced_already and not args.intervention_other_bidirectionalspec:
                                # These are strictly self roles
                                padded_add_roles += [candidate_role]
                                padded_add_fillers += [this_f_replaced]




                for pair_r, pair_f in pairs_no_pad: # Iterate over the pairs in the current token

                    if args.intervention_self_bidirectionalspec:
                        # These are exclusively non-self changes
                        continue

                    # pair_r_r1 = earlier token's role
                    # pair_r_r2 = current token's role
                    pair_r_r1, pair_r_r2 = split_role(pair_r)
                    if pair_r_r1 == r_to_replace and pair_f == f_to_replace and pair_r != "0_" + r_to_replace + "-" + r_to_replace :
                        # A preceding token contains a role-filler pair that should be replaced
                        # (Not the current token - that's ruled out by pair_r != "0_" + r_to_replace + "-" + r_to_replace)
                        # (This is why this part of the code is only non-self replacements)

                        if r_to_use != "DUMMY":

                            # The role that would be created by changing the relevant preceding role-filler pair
                            candidate_first_part = r_to_use
                            candidate_second_part = pair_r_r2

                            if replacement_current_token_role is not None:
                                candidate_first_part = r_to_use.replace(current_token_role, replacement_current_token_role)
                                candidate_second_part = pair_r_r2.replace(current_token_role, replacement_current_token_role)
                            candidate_role = "0_" + candidate_first_part + "-" + candidate_second_part

                            doubled = (candidate_first_part == candidate_second_part)


                            if not doubled and candidate_role in tpe_ensemble.tpe_3.seq2roles.role_token2idx:
                                # Check if the candidate role exists as a possible role
                                # If so, it means that it is expected to be present in the current token,
                                # so we should replace the old pair with the candidate one, assuming
                                # we haven't already added it
                                if (candidate_role, f_to_use) not in list(zip(padded_add_roles, padded_add_fillers)):
                                    padded_add_roles += [candidate_role]
                                    padded_subtract_roles += [pair_r]
                                    padded_add_fillers += [f_to_use]
                                    padded_subtract_fillers += [f_to_replace]



                            else:
                                # If it doesn't exist, this means that the replacement we are doing
                                # moves a preceding token in such a way that it no longer is part
                                # of the representation of the current token. Thus, we only need
                                # to subtract it, not add it.
                                if (pair_r, f_to_replace) not in list(zip(padded_subtract_roles, padded_subtract_fillers)):
                                    padded_subtract_roles += [pair_r]
                                    padded_subtract_fillers += [f_to_replace]


                        else:
                            # If the role is "DUMMY", we use whatever role it already had and just change the filler
                            padded_subtract_fillers += [f_to_replace]
                            padded_subtract_roles += [pair_r]
                            padded_add_fillers += [f_to_use]
                            padded_add_roles += [pair_r]



                # A new candidate of a role to create from scratch
                extra_candidate = "0_" + r_to_use + "-" + current_token_role
                if replacement_current_token_role is not None:
                    # If the candidate role involves a role that will be changed by our edits,
                    # we need to change that role in the candidate
                    extra_candidate = extra_candidate.replace(current_token_role, replacement_current_token_role)


                if extra_candidate in tpe_ensemble.tpe_3.seq2roles.role_token2idx:
                    # If this new candidate is a possible role (above line) and has not yet been added and is not going to be
                    # added in a new replacement role (below line), then we add it in
                    if (extra_candidate, f_to_use) not in list(zip(padded_add_roles, padded_add_fillers)) and current_token_role not in all_roles_to_replace:
                        
                        if not args.intervention_self_bidirectionalspec:
                            # This is strictly a non-self role
                            padded_add_roles += [extra_candidate]
                            padded_add_fillers += [f_to_use]


            fr_string = get_fr_string(fillers, roles, padded_add_fillers, padded_add_roles, padded_subtract_fillers, padded_subtract_roles)

        else:

            for fr_replacement in fr_replacements_processed:
                all_roles_present = True
                found_pair = False
                for r, f in fr_replacement[0]:
                    for pair_r, pair_f in pairs:
                        if pair_r == r and pair_f == f:
                            found_pair = True

                    if not found_pair:
                        all_roles_present = False

                
                if all_roles_present:

                    padded_subtract_fillers += [elt[1] for elt in fr_replacement[0]]
                    padded_subtract_roles += [elt[0] for elt in fr_replacement[0]]
                    padded_add_fillers += [elt[1] for elt in fr_replacement[1]]
                    padded_add_roles += [elt[0] for elt in fr_replacement[1]]



        while len(padded_subtract_fillers) < max_n_replacements:
            padded_subtract_fillers.append("<PAD>")
        fillers_to_subtract[fr_index] = [tpe_ensemble.tpe_3.seq2roles.filler_token2idx[elt] for elt in padded_subtract_fillers]

        while len(padded_subtract_roles) < max_n_replacements:
            padded_subtract_roles.append("<PAD>")
        roles_to_subtract[fr_index] = [tpe_ensemble.tpe_3.seq2roles.role_token2idx[elt] for elt in padded_subtract_roles]


        while len(padded_add_fillers) < max_n_replacements:
            padded_add_fillers.append("<PAD>")
        fillers_to_add[fr_index] = [tpe_ensemble.tpe_3.seq2roles.filler_token2idx[elt] for elt in padded_add_fillers]

        while len(padded_add_roles) < max_n_replacements:
            padded_add_roles.append("<PAD>")
        roles_to_add[fr_index] = [tpe_ensemble.tpe_3.seq2roles.role_token2idx[elt] for elt in padded_add_roles]
                       

    # 25 because that's how many layers are in GPT-OSS
    batch_to_subtract = [{"fillers":fillers_to_subtract, "roles":roles_to_subtract}]*25
    batch_to_add = [{"fillers":fillers_to_add, "roles":roles_to_add}]*25

    _, tpe_to_subtract = tpe_ensemble(batch_to_subtract)
    _, tpe_to_add = tpe_ensemble(batch_to_add)

    _, hidden_list = tpe_ensemble(examples)
    _, hidden_list_eval = tpe_ensemble(examples_eval)
    
    if intervention_set.mean_vector_list is not None:
        hidden_list = [(hidden*intervention_set.stddev_vector_list[layer_index]) + intervention_set.mean_vector_list[layer_index] for layer_index, hidden in enumerate(hidden_list)]
        hidden_list_eval = [(hidden*intervention_set.stddev_vector_list[layer_index]) + intervention_set.mean_vector_list[layer_index] for layer_index, hidden in enumerate(hidden_list_eval)]
    else:
        hidden_list = [hidden for layer_index, hidden in enumerate(hidden_list)]
        hidden_list_eval = [hidden for layer_index, hidden in enumerate(hidden_list_eval)] 



    llm_string = tokenizer.batch_decode([examples[0]["input_seq"][-1]], skip_special_tokens=False)[0]
    model_inputs = tokenizer([llm_string], return_tensors="pt").to(llm_model.device)

    # We assume there's only one example in the batch (because
    # swapping accuracy only works in that case)
    correct_answer = examples[0]["answer"][0]

    m_outputs = llm_model(**model_inputs, output_hidden_states=True, use_cache=True)
    m_hidden_orig = m_outputs.hidden_states
    answer_hidden, _ = generate_text_from_hidden(tokenizer, model_inputs, llm_model, m_hidden_orig, truncate_token=args.truncate_token, max_length=args.max_length)
    solution_hidden = postprocess_llm_answer(answer_hidden, args.prefix_str_length, truncate_before_strs=args.truncate_before_strs, truncate_after_strs=args.truncate_after_strs, delete_strings=delete_strings)


    llm_string_eval = tokenizer.batch_decode([examples_eval[0]["input_seq"][-1]], skip_special_tokens=False)[0]
    model_inputs_eval = tokenizer([llm_string_eval], return_tensors="pt").to(llm_model.device)

    # We assume there's only one example in the batch (because
    # swapping accuracy only works in that case)
    correct_answer_eval = examples_eval[0]["answer"][0]

    m_outputs = llm_model(**model_inputs_eval, output_hidden_states=True, use_cache=True)
    m_hidden = m_outputs.hidden_states
    answer_hidden_eval, _ = generate_text_from_hidden(tokenizer, model_inputs_eval, llm_model, m_hidden, truncate_token=args.truncate_token, max_length=args.max_length)
    solution_hidden_eval = postprocess_llm_answer(answer_hidden_eval, args.prefix_str_length, truncate_before_strs=args.truncate_before_strs, truncate_after_strs=args.truncate_after_strs, delete_strings=delete_strings)



    if examples_foil is not None:

        llm_string_foil = tokenizer.batch_decode([examples_foil[0]["input_seq"][-1]], skip_special_tokens=False)[0]
        model_inputs_foil = tokenizer([llm_string_foil], return_tensors="pt").to(llm_model.device)

        # We assume there's only one example in the batch (because
        # swapping accuracy only works in that case)
        correct_answer_foil = examples_foil[0]["answer"][0]

        m_outputs = llm_model(**model_inputs_foil, output_hidden_states=True, use_cache=True)
        m_hidden_foil = m_outputs.hidden_states
        answer_hidden_foil, _ = generate_text_from_hidden(tokenizer, model_inputs_foil, llm_model, m_hidden, truncate_token=args.truncate_token, max_length=args.max_length)
        solution_hidden_foil = postprocess_llm_answer(answer_hidden_foil, args.prefix_str_length, truncate_before_strs=args.truncate_before_strs, truncate_after_strs=args.truncate_after_strs, delete_strings=delete_strings)


        llm_string_foil_eval = tokenizer.batch_decode([examples_foil_eval[0]["input_seq"][-1]], skip_special_tokens=False)[0]
        model_inputs_foil_eval = tokenizer([llm_string_foil_eval], return_tensors="pt").to(llm_model.device)

        # We assume there's only one example in the batch (because
        # swapping accuracy only works in that case)
        correct_answer_foil_eval = examples_foil_eval[0]["answer"][0]

        m_outputs = llm_model(**model_inputs_foil_eval, output_hidden_states=True, use_cache=True)
        m_hidden = m_outputs.hidden_states
        answer_hidden_foil_eval, _ = generate_text_from_hidden(tokenizer, model_inputs_foil_eval, llm_model, m_hidden, truncate_token=args.truncate_token, max_length=args.max_length)
        solution_hidden_foil_eval = postprocess_llm_answer(answer_hidden_foil_eval, args.prefix_str_length, truncate_before_strs=args.truncate_before_strs, truncate_after_strs=args.truncate_after_strs, delete_strings=delete_strings)






    if args.suffix_length is not None:
        hidden_modified = [x.squeeze(0)[:-1*args.suffix_length] for x in m_hidden_orig]
        if examples_foil is not None:
            hidden_foil_modified = [x.squeeze(0)[:-1*args.suffix_length] for x in m_hidden_foil]
    else:
        hidden_modified = [x.squeeze(0)[:args.prefix_length] for x in m_hidden_orig]
        if examples_foil is not None:
            hidden_foil_modified = [x.squeeze(0)[:args.prefix_length] for x in m_hidden_foil] 


    encoding_list = [examples[example_index]["encoding"] for example_index in range(len(examples))]
    if intervention_set.mean_vector_list is not None:
        encoding_list = [hidden for layer_index, hidden in enumerate(encoding_list)]
        tpe_to_subtract = [(hidden*intervention_set.stddev_vector_list[layer_index]) + intervention_set.mean_vector_list[layer_index] for layer_index, hidden in enumerate(tpe_to_subtract)]
        tpe_to_add = [(hidden*intervention_set.stddev_vector_list[layer_index]) + intervention_set.mean_vector_list[layer_index] for layer_index, hidden in enumerate(tpe_to_add)]
    else:
        encoding_list = [hidden for layer_index, hidden in enumerate(encoding_list)]
        tpe_to_subtract = [hidden for layer_index, hidden in enumerate(tpe_to_subtract)] 
        tpe_to_add = [hidden for layer_index, hidden in enumerate(tpe_to_add)]


    if examples_foil is not None:
        encoding_list_foil = [examples_foil[example_index]["encoding"] for example_index in range(len(examples_foil))]
        if intervention_foil_set.mean_vector_list is not None:
            encoding_list_foil = [hidden for layer_index, hidden in enumerate(encoding_list_foil)]
        else:
            encoding_list_foil = [hidden for layer_index, hidden in enumerate(encoding_list_foil)]


    hidden_list_intervention = [hidden.clone().detach() for layer_index, hidden in enumerate(hidden_list)]
    hidden_list_replace = [hidden.clone().detach() for layer_index, hidden in enumerate(hidden_list)]
    hidden_list = [x.to(dtype=torch.bfloat16).squeeze(0) for x in hidden_list]
    hidden_list_eval = [x.to(dtype=torch.bfloat16).squeeze(0) for x in hidden_list_eval]
    hidden_combined = [torch.cat([hidden_modified[i], hidden_list[i]], 0).unsqueeze(0) for i in range(len(hidden_list))]
    hidden_combined_eval = [torch.cat([hidden_modified[i], hidden_list_eval[i]], 0).unsqueeze(0) for i in range(len(hidden_list_eval))]

    answer_rec, _ = generate_text_from_hidden(tokenizer, model_inputs, llm_model, hidden_combined, truncate_token=args.truncate_token, max_length=args.max_length)
    solution_rec = postprocess_llm_answer(answer_rec, args.prefix_str_length, truncate_before_strs=args.truncate_before_strs, truncate_after_strs=args.truncate_after_strs, delete_strings=delete_strings)


    answer_rec_eval, _ = generate_text_from_hidden(tokenizer, model_inputs_eval, llm_model, hidden_combined_eval, truncate_token=args.truncate_token, max_length=args.max_length)
    solution_rec_eval = postprocess_llm_answer(answer_rec_eval, args.prefix_str_length, truncate_before_strs=args.truncate_before_strs, truncate_after_strs=args.truncate_after_strs, delete_strings=delete_strings)


    for encoding_index in range(len(encoding_list)):
        encoding_list[encoding_index] = encoding_list[encoding_index] - tpe_to_subtract[encoding_index] + tpe_to_add[encoding_index]
        hidden_list_intervention[encoding_index] = hidden_list_intervention[encoding_index] - tpe_to_subtract[encoding_index] + tpe_to_add[encoding_index]
      
        if examples_foil is not None:
            encoding_list_foil[encoding_index] = encoding_list_foil[encoding_index] - tpe_to_subtract[encoding_index] + tpe_to_add[encoding_index]

    
    if args.intervention_self_bidirectional or args.intervention_self_bidirectionalspec:
        for encoding_index in range(len(encoding_list)):
            for index_to_replace in indices_to_replace:
                hidden_list_replace[encoding_index][0][index_to_replace] = tpe_to_add[encoding_index][0][index_to_replace]
        hidden_list_replace = [x.to(dtype=torch.bfloat16).squeeze(0) for x in hidden_list_replace]
        encoding_combined_replace = [torch.cat([hidden_modified[i], hidden_list_replace[i]], 0).unsqueeze(0) for i in range(len(encoding_list))]

        answer_replace, _ = generate_text_from_hidden(tokenizer, model_inputs, llm_model, encoding_combined_replace, truncate_token=args.truncate_token, max_length=args.max_length)
        solution_replace = postprocess_llm_answer(answer_replace, args.prefix_str_length, truncate_before_strs=args.truncate_before_strs, truncate_after_strs=args.truncate_after_strs, delete_strings=delete_strings)



    hidden_list_intervention = [x.to(dtype=torch.bfloat16).squeeze(0) for x in hidden_list_intervention]

    encoding_list = [x.to(dtype=torch.bfloat16).squeeze(0) for x in encoding_list]
    encoding_combined_intervention = [torch.cat([hidden_modified[i], hidden_list_intervention[i]], 0).unsqueeze(0) for i in range(len(encoding_list))]
    encoding_combined = [torch.cat([hidden_modified[i], encoding_list[i]], 0).unsqueeze(0) for i in range(len(encoding_list))]
    answer_encoding, _ = generate_text_from_hidden(tokenizer, model_inputs, llm_model, encoding_combined, truncate_token=args.truncate_token, max_length=args.max_length)
    solution_encoding = postprocess_llm_answer(answer_encoding, args.prefix_str_length, truncate_before_strs=args.truncate_before_strs, truncate_after_strs=args.truncate_after_strs, delete_strings=delete_strings)

    if examples_foil is not None:
        encoding_list_foil = [x.to(dtype=torch.bfloat16).squeeze(0) for x in encoding_list_foil]
        encoding_combined_foil = [torch.cat([hidden_foil_modified[i], encoding_list_foil[i]], 0).unsqueeze(0) for i in range(len(encoding_list_foil))]
        answer_encoding_foil, _ = generate_text_from_hidden(tokenizer, model_inputs_foil, llm_model, encoding_combined_foil, truncate_token=args.truncate_token, max_length=args.max_length)
        solution_encoding_foil = postprocess_llm_answer(answer_encoding_foil, args.prefix_str_length, truncate_before_strs=args.truncate_before_strs, truncate_after_strs=args.truncate_after_strs, delete_strings=delete_strings)
        correct_new_answer_foil = examples_foil[0]["new_answer"][0]


    answer_tpe_encoding, _ = generate_text_from_hidden(tokenizer, model_inputs, llm_model, encoding_combined_intervention, truncate_token=args.truncate_token, max_length=args.max_length)
    solution_tpe_encoding = postprocess_llm_answer(answer_tpe_encoding, args.prefix_str_length, truncate_before_strs=args.truncate_before_strs, truncate_after_strs=args.truncate_after_strs, delete_strings=delete_strings)

    correct_new_answer = examples[0]["new_answer"][0]

    logging.info(llm_string)
    logging.info("LLM orig:          " + solution_hidden) # The solution that the LLM gives when provided its own unmodified hidden states for the original prompt
    logging.info("TPE orig:          " + solution_rec) # The solution that the LLM gives when provided TPE approximations for its own unmodified hidden states for the original prompt
    logging.info("Correct orig:      " + correct_answer) # Correct answer for the original prompt
    logging.info("LLM orig eval:     " + solution_hidden_eval) # The solution that the LLM gives when provided its own unmodified hidden states for the prompt that should result from the intervention
    logging.info("TPE eval:          " + solution_rec_eval) # The solution that the LLM gives when provided TPE approximations for its own unmodified hidden states for the prompt that should result from the intervention
    if args.intervention_self_bidirectional or args.intervention_self_bidirectionalspec:
        logging.info("Replace:           " + solution_replace) # Answer produced by starting with LLM encoding and entirely replacing it with the "add" TPE pairs
    logging.info("TPE enc solution:  " + solution_tpe_encoding) # Answer produced by starting with TPE encoding and adding/subtracting targeted TPE pairs
    if examples_foil is not None:
        logging.info("LLM foil:          " + solution_hidden_foil) # The solution that the LLM gives when provided its own unmodified hidden states for the original prompt for the foil example
        logging.info("Correct orig foil: " + correct_answer_foil) # Correct answer for the foil prompt
        logging.info("LLM foil eval:     " + solution_hidden_foil_eval) # The solution that the LLM gives when provided its own unmodified hidden states for the prompt that should result from conducting the intervention on the foil example
        logging.info("Enc soln foil:     " + solution_encoding_foil) # Answer produced by starting with LLM encoding for the foil prompt and adding/subtracting targeted TPE pairs
        logging.info("Correct foil:      " + correct_new_answer_foil) # Correct answer for conducting an intervention on the foil prompt
    logging.info("Enc solution:      " + solution_encoding) # Answer produced by starting with LLM encoding and adding/subtracting targeted TPE pairs
    logging.info("Correct:           " + correct_new_answer) # What the post-intervention answer should be
    logging.info("")


    total_interventions += 1
    if args.intervention_self_bidirectional or args.intervention_self_bidirectionalspec:
        if solution_replace == correct_new_answer:
            total_replace_correct += 1
        if solution_replace == solution_hidden_eval:
            total_replace_agreement += 1
    if solution_encoding == correct_new_answer:
        total_interventions_correct += 1
    if solution_encoding == solution_hidden_eval:
        total_interventions_agreement += 1
    if solution_tpe_encoding == correct_new_answer:
        total_tpe_interventions_correct += 1
    if solution_tpe_encoding == solution_hidden_eval:
        total_tpe_interventions_agreement += 1
    if solution_hidden == correct_answer:
        total_llm_correct += 1
    if solution_hidden_eval == correct_answer_eval:
        total_llm_correct_eval += 1
    if solution_rec == correct_answer:
        total_tpe_correct += 1
    if solution_rec == solution_hidden:
        total_tpe_agreement += 1
    if solution_rec_eval == correct_new_answer:
        total_tpe_correct_eval += 1
    if solution_rec_eval == solution_hidden_eval:
        total_tpe_agreement_eval += 1

    if examples_foil is not None:
        if solution_hidden_foil == correct_answer_foil:
            total_llm_correct_foil += 1
        if solution_hidden_foil_eval == correct_answer_foil_eval:
            total_llm_correct_eval_foil += 1
        if solution_encoding_foil == correct_new_answer_foil:
            total_interventions_correct_foil += 1
        if solution_encoding_foil == solution_hidden_foil_eval:
            total_interventions_agreement_foil += 1


logging.info("Test target acc:       " + str(total_llm_correct*1.0/total_interventions)) # % LLM correct on original prompts
logging.info("Test accuracy:         " + str(total_tpe_correct*1.0/total_interventions)) # % of TPE approximation of the original prompts that yield the right answer
logging.info("Test agreement:        " + str(total_tpe_agreement*1.0/total_interventions)) # % of TPE approximation of the original prompts that yield the same answer as the LLM
logging.info("Test target acc eval:  " + str(total_llm_correct_eval*1.0/total_interventions)) # % LLM correct on prompts that should result from the interventions
logging.info("Test TPE acc eval:     " + str(total_tpe_correct_eval*1.0/total_interventions)) # % of TPE approximation of the modified prompts that yield the right answer
logging.info("Test TPE agr eval:     " + str(total_tpe_agreement_eval*1.0/total_interventions)) # % of TPE approximation of the modified prompts that yield the same answer as the LLM
if args.intervention_self_bidirectional or args.intervention_self_bidirectionalspec:
    logging.info("Replace acc:           " + str(total_replace_correct*1.0/total_interventions)) # % of correct answers from intervening by entirely replacing the LLM hidden states with the "add" TPEs
    logging.info("Replace agr:           " + str(total_replace_agreement*1.0/total_interventions)) # % of answers where replacing the LLM hidden states with the "add" TPEs gives the same answer as feeding the LLM the modified prompt
logging.info("TPE intervention acc:  " + str(total_tpe_interventions_correct*1.0/total_interventions)) # % of interventions that succeed when starting with the TPE encoding
logging.info("TPE intervention agr:  " + str(total_tpe_interventions_agreement*1.0/total_interventions)) # % of cases where the intervention gives the same answer as feeding the LLM hidden states for the target prompt
if args.intervention_foil is not None:
    logging.info("Foil target acc:       " + str(total_llm_correct_foil*1.0/total_interventions)) # % LLM correct on original prompts for foil examples
    logging.info("Foil target acc eval : " + str(total_llm_correct_eval_foil*1.0/total_interventions)) # % LLM correct on original prompts for the modified foil examples
logging.info("Intervention acc:      " + str(total_interventions_correct*1.0/total_interventions)) # % of LLM-modified-by-TPE interventions that succeed
logging.info("Intervention agr:      " + str(total_interventions_agreement*1.0/total_interventions)) # % LLM answers that are the same when given intervened original states vs. its own hidden states for the intervened prompts
if args.intervention_foil is not None:
    logging.info("Intervention acc foil: " + str(total_interventions_correct_foil*1.0/total_interventions)) # % of LLM-modified-by-TPE interventions that succeed for the foil cases
    logging.info("Intervention agr foil: " + str(total_interventions_agreement_foil*1.0/total_interventions)) # % LLM answers that are the same when given intervened original states vs. its own hidden states for the intervened prompts for the foil cases



