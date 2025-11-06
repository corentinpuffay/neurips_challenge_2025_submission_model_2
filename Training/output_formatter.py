import scipy as sp

from settings import Settings

settings = Settings(challenge=2)

input_filepath = settings.log_path()
output_filepath, summary_filepath = settings.summary_paths()

outfile = open(output_filepath, 'w')
summary_file = open(summary_filepath, 'w')

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
for combination in data:
    outfile.write(f"{combination}\n")
    summary_file.write(f"{combination}\n")
    outfile.write(f"Settings: {data[combination]['settings']}\n")

    # Write correlation between val_loss and test_loss
    if len(data[combination]['val_loss']) > 0 and len(data[combination]['test_loss']) > 0:
        min_losses = data[combination]['min_loss']
        max_losses = data[combination]['max_loss']
        val_losses = data[combination]['val_loss']
        test_losses = data[combination]['test_loss']

        correlation = sp.stats.spearmanr(min_losses, test_losses).statistic
        outfile.write(f"Correlation between min loss and test loss: {correlation:.4f}\n")

        correlation = sp.stats.spearmanr(max_losses, test_losses).statistic
        outfile.write(f"Correlation between max loss and test loss: {correlation:.4f}\n")

        correlation = sp.stats.spearmanr(val_losses, test_losses).statistic
        outfile.write(f"Correlation between validation loss and test loss: {correlation:.4f}\n")

    # Print test loss at the epoch with the lowest validation loss
    if len(data[combination]['val_loss']) > 0 and len(data[combination]['test_loss']) > 0:
        min_val_loss = min(data[combination]['val_loss'])
        min_val_loss_index = data[combination]['val_loss'].index(min_val_loss)
        min_val_loss_epoch = data[combination]['epoch'][min_val_loss_index]
        corresponding_test_loss = data[combination]['test_loss'][min_val_loss_index]
        outfile.write(f"Test loss at epoch {min_val_loss_epoch} with lowest validation loss ({min_val_loss:.4f}): {corresponding_test_loss:.4f}\n")
        summary_file.write(f"Test loss at epoch {min_val_loss_epoch} with lowest validation loss ({min_val_loss:.4f}): {corresponding_test_loss:.4f}\n")
    
    outfile.write("Epoch\tMin Loss\tMax Loss\tVal Loss\tTest Loss\tLearning Rate\n")
    for i in range(len(data[combination]['epoch'])):
        epoch = data[combination]['epoch'][i]
        min_loss = data[combination]['min_loss'][i]
        max_loss = data[combination]['max_loss'][i]
        val_loss = data[combination]['val_loss'][i] if i < len(data[combination]['val_loss']) else 'N/A'
        test_loss = data[combination]['test_loss'][i] if i < len(data[combination]['test_loss']) else 'N/A'
        learning_rate = data[combination]['learning_rate'][i]
        outfile.write(f"{epoch}\t{min_loss:.4f}\t{max_loss:.4f}\t{val_loss}\t{test_loss}\t{learning_rate:.6f}\n")
    outfile.write("\n")