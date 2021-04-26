def run_save_load_train(happy, output_path, args, data_path, run_type):
    args['save_data_path'] = output_path
    args['load_data_path'] = output_path

    args['save_data'] = True
    if run_type == "train":
        happy.train(data_path, args=args)
    elif run_type == "eval":
        result = happy.eval(data_path, args=args)
    elif run_type == "test":
        result = happy.test(data_path, args=args)
    else:
        print("WARNING: invalid run_type for run_save_load_train")
        return
    args['save_data'] = False

    args['load_data'] = True
    if run_type == "train":
        happy.train(data_path, args=args)
    elif run_type == "eval":
        result = happy.eval(data_path, args=args)
    elif run_type == "test":
        result = happy.test(data_path, args=args)

    args['save_data'] = False



