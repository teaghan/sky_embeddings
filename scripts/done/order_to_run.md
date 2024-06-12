1. generate_h5.sh --> to generate validation, redshifts, dwarfs sets (need to change python code being called)
2. dwarf_class.sh --> merge some of validation in with the dwarf class to have 50% dwarfs 50% not
3. mim_unions.sh --> to train the model (give GPU too)
4. mim_unions_zspec_train.sh --> train linear probe model (give GPU ideally)
5. mim_unions_zspec_test.sh --> test linear probe model (give GPU ideally)
6. dwarf_seach.sh (or dwarf_search_single for just one target) --> similarity search (give GPU ideally and can be run before linear probe)