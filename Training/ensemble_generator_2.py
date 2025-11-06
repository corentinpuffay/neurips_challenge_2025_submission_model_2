import os
import uuid
import shutil
from multiprocessing import freeze_support
from datetime import datetime

from main_transformer_contrastive import train
from settings import Settings
from itertools import product

def do_run(f, settings):
    f.write(str(settings))
    train(settings, f)

def format_output(settings):
    input_filepath = settings.ensemble_log_path()
    summary_filepath = settings.ensemble_summary_path()
    summary_file = open(summary_filepath, 'a')

    data = {}

    # Read the content of the input file
    with open(input_filepath, 'r') as infile:
        lines = infile.readlines()

        # Search for lines containing "=== Combination x ===" with x being an integer
        combination_lines = [line for line in lines if "=== Combination" in line]
        start_line = True
        for line in combination_lines:
            if start_line:
                start_line = False
            else:
                start_line = True

            # Check if line is a key in data dictionary
            if line not in data:
                line = line.replace("=", "")
                line = line.strip()
                data[line] = {'settings': None, 'epoch': [], 'min_loss': [], 'max_loss': [], 'val_loss': [], 'test_loss': [], 'learning_rate': []}

    # Read the content of the input file
    validation_type = 0
    with open(input_filepath, 'r') as infile:
        # Read line by line
        for line in infile:
            if "=== Combination" in line:
                current_combination = line.replace("=", "")
                current_combination = current_combination.strip()
            elif "Settings" in line:
                settings = line.replace('EEGTransformer(', '')
                data[current_combination]["settings"] = settings.strip()
            elif "Epoch" in line:
                # Get epoch, min_loss, max_loss and learning rate from the line, e.g.: Epoch 0 took 9.94 seconds - min loss: 0.9804 - max loss: 3.5970 - lr: 0.010000
                parts = line.strip().split(" took ")
                epoch = int(parts[0].split(" ")[1])
                parts = line.strip().split(" - ")
                min_loss = float(parts[1].split(": ")[1])
                max_loss = float(parts[2].split(": ")[1])
                learning_rate = float(parts[3].split(": ")[1])

                data[current_combination]['epoch'].append(epoch)
                data[current_combination]['min_loss'].append(min_loss)
                data[current_combination]['max_loss'].append(max_loss)
                data[current_combination]['learning_rate'].append(learning_rate)
            elif "Validation loss after epoch" in line:
                # Get validation loss from the line, e.g.: Validation loss after epoch 0: 0.9234
                parts = line.strip().split(": ")
                val_loss = float(parts[1])
                if validation_type == 0:
                    data[current_combination]['val_loss'].append(val_loss)
                    validation_type += 1
                elif validation_type == 1:
                    validation_type += 1
                else:
                    data[current_combination]['test_loss'].append(val_loss)
                    validation_type = 0

    # Write formatted output to the output file
    best_models = {}
    for combination in data:
        summary_file.write(f"{combination}\n")

        # Print test loss at the epoch with the lowest validation loss
        if len(data[combination]['val_loss']) > 0 and len(data[combination]['test_loss']) > 0:
            min_val_loss = min(data[combination]['val_loss'])
            min_val_loss_index = data[combination]['val_loss'].index(min_val_loss)
            min_val_loss_epoch = data[combination]['epoch'][min_val_loss_index]
            best_models[combination] = min_val_loss_epoch
            corresponding_test_loss = data[combination]['test_loss'][min_val_loss_index]
            summary_file.write(f"Test loss at epoch {min_val_loss_epoch} with lowest validation loss ({min_val_loss:.4f}): {corresponding_test_loss:.4f}\n")

    summary_file.close()
    return best_models

def copy_best_models(settings):
    # Run output_formatter
    best_models = format_output(settings)
    print(best_models)

    # Copy best models
    model_folder = settings.model_folder()
    target_folder = settings.good_model_folder()
    for combination in best_models:
        epoch = best_models[combination]
        combination = combination.replace("Combination ", "")
        filepath = os.path.join(model_folder, f"model_weights_{combination}_{epoch}.pt")
        model_uuid = uuid.uuid1()
        target_path = os.path.join(target_folder, f"model_weights_{combination}_{epoch}_{model_uuid}.pt")
        shutil.copyfile(filepath, target_path)

    # Remove other models
    shutil.rmtree(model_folder)

    # Make new model folder
    os.makedirs(model_folder)

if __name__ == '__main__':
    freeze_support()

    interesting_combos = [11]
    # interesting_combos = [26, 32, 39]

    while True:
        for current_combo in interesting_combos:
            settings = Settings(challenge=2)

            f = open(settings.ensemble_log_path(), 'w')
            f.write(f"Run started at {datetime.now().isoformat()}\n")

            for run in range(2):
                # Define the hyperparameter grid
                d_models = [16, 64, 256]
                dim_feedforwards = [2, 4]
                num_layers = [1, 3]
                nheads = [1, 4]
                dropouts = [0.1, 0.3]
                huber_deltas = [0.5, 1.0, 2.0]
                loss_fns = ['huber', 'NRMSE']

                combo_nb = 0
                for combo in product(d_models, dim_feedforwards, num_layers, nheads, dropouts, huber_deltas, loss_fns):

                    # If not the current combo, skip
                    if combo_nb != current_combo:
                        combo_nb += 1
                        continue

                    settings.d_model = combo[0]
                    settings.dim_feedforward = combo[1] * settings.d_model
                    settings.num_layers = combo[2]
                    settings.nheads = combo[3]
                    settings.dropout = combo[4]
                    settings.huber_delta = combo[5]
                    settings.loss_fn = combo[6]

                    f.write(f"\n=== Combination {combo_nb} ===\n")
                    settings.number = combo_nb
                    do_run(f, settings)
                    f.write(f"\n=== Combination {combo_nb} ===\n\n\n\n")
                    combo_nb += 1
            f.close()
            copy_best_models(settings)
