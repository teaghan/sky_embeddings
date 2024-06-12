# for nick

1. generate_h5.sh --> to generate validation, redshifts, dwarfs sets (need to change python code being called)
2. dwarf_class.sh --> merge some of validation in with the dwarf class to have 50% dwarfs 50% not
3. mim_unions.sh --> to train the model (give GPU too)*
4. mim_unions_zspec_train.sh --> train linear/fine-tune probe model (give GPU ideally)
5. mim_unions_zspec_test.sh --> test linear probe model (give GPU ideally)
6. dwarf_seach.sh (or dwarf_search_single for just one target) --> similarity search (give GPU ideally and can be run before linear probe)

```
sbatch --time=00:20:00 --cpus-per-task=6 --account=rrg-kyi --mem=100000M --gres=gpu:1 ./scripts/done/mim_unions.sh
```

*if submitting big job can run the following instead of mim_unions.sh if you want to submit the big job as multiple smaller jobs:
```
python cc/queue_cc.py --account "def-sfabbro" --todo_dir "scripts/todo/" --done_dir "scripts/done/" --output_dir "scripts/stdout/" --num_jobs 1 --num_runs 3 --num_gpu 1 --num_cpu 6 --mem 10G --time_limit "00-03:00"
```
you can see all these jobs when you sq but all of them after the first will show (Dependency) next to them as they will wait for the first one to complete. you also need to make sure this is hooked up to a streaming dataloader that either pulls tiles randomly or keeps track of what has been pulled already.

other notes:
- need to change all /home/a4ferrei to your home
- maybe change my env folder to your env
- change github folder to where your code is
- currently valiation set is being used for training and testing, change this when doing real runs