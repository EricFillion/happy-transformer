def run_save_load(happy, output_path, args, data_path, run_type):
    args.save_path = output_path
    args.load_path = False

    if run_type == "train":
        happy.train(data_path, args=args)
    elif run_type == "eval":
        result = happy.eval(data_path, args=args)
    elif run_type == "test":
        result = happy.test(data_path, args=args)
    else:
        ValueError("invalid run_type for run_save_load")

    args.save_path = False
    args.load_path = output_path

    if run_type == "train":
        happy.train(data_path, args=args)
    elif run_type == "eval":
        result = happy.eval(data_path, args=args)
    elif run_type == "test":
        result = happy.test(data_path, args=args)


