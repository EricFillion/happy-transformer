from happytransformer.happy_trainer import action_step

def test_action_step():
    ape = 1
    batch_size = 1
    gas= 1
    data_len= 1
    num_gpus= 1

    test_cases =[{"ape": 1,
                 "batch_size": 1,
                 "gas": 1,
                 "data_len": 1,
                 "num_gpus": 1,
                 "result": 1},

                 {"ape": 1,
                  "batch_size": 1,
                  "gas": 1,
                  "data_len": 2,
                  "num_gpus": 1,
                  "result": 2},

                 {"ape": 2,
                  "batch_size": 1,
                  "gas": 1,
                  "data_len": 16,
                  "num_gpus": 1,
                  "result": 8},

                 {"ape": 1,
                  "batch_size": 2,
                  "gas": 1,
                  "data_len": 16,
                  "num_gpus": 1,
                  "result": 8},

                 {"ape": 1,
                  "batch_size": 1,
                  "gas": 2,
                  "data_len": 16,
                  "num_gpus": 1,
                  "result": 8},

                 {"ape": 1,
                  "batch_size": 2,
                  "gas": 2,
                  "data_len": 16,
                  "num_gpus": 1,
                  "result": 4},

                 {"ape": 2,
                  "batch_size": 2,
                  "gas": 2,
                  "data_len": 16,
                  "num_gpus": 1,
                  "result": 2},

                 {"ape": 2,
                  "batch_size": 2,
                  "gas": 2,
                  "data_len": 16,
                  "num_gpus": 2,
                  "result": 1},
                 ]

    for case in test_cases:
        out = action_step(
            ape=case["ape"],
            batch_size=case["batch_size"],
            gas=case["gas"],
            data_len=case["data_len"],
            num_gpus=case["num_gpus"])

        assert out == case["result"]