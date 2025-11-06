from multiprocessing import freeze_support
from datetime import datetime

from main_transformer_contrastive import train
# from main_transformer import train
from settings import Settings
from itertools import product

def do_run(f, settings):
    f.write(str(settings))
    train(settings, f)

if __name__ == '__main__':
    freeze_support()

    # skip_combo_till = -1  # for resuming interrupted sweeps (-1 means no skipping)
    interesting_combos = [26, 32, 39, 0, 1, 2, 3, 4, 6, 7, 8, 10, 11, 20, 24, 25, 27, 28, 29, 30, 33, 34, 36, 38]
    interesting_combos = [0, 26, 32, 39]
    interesting_combos = [39]
    interesting_combos = [11, 35]

    for run in range(2):
        # Define the hyperparameter grid
        d_models = [16, 64, 256]
        dim_feedforwards = [2, 4]
        num_layers = [1, 3]
        nheads = [1, 4]
        dropouts = [0.1, 0.3]
        huber_deltas = [0.5, 1.0, 2.0]
        loss_fns = ['huber', 'NRMSE']

        settings = Settings(challenge=2)
        f = open(settings.log_path(), 'a')
        f.write(f"Run started at {datetime.now().isoformat()}\n")

        combo_nb = 0
        for combo in product(d_models, dim_feedforwards, num_layers, nheads, dropouts, huber_deltas, loss_fns):
            # if combo_nb < skip_combo_till:
            #     combo_nb += 1
            #     continue

            # If not in the interesting combos, skip
            if combo_nb not in interesting_combos:
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