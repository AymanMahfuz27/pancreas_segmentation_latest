import fileinput

filename = 'train_3d.py'
search_text = 'loss = function.train_sam(args, net, optimizer, nice_train_loader, epoch)'
replace_text = 'loss = function.train_sam(args, net, optimizer, None, nice_train_loader, epoch)'

with fileinput.FileInput(filename, inplace=True, backup='.bak') as file:
    for line in file:
        print(line.replace(search_text, replace_text), end='')
