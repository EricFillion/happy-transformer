---
layout: page
title: WandB Tracking
permalink: /wandb/
nav_order: 17
---
You can use [WandB](https://wandb.ai/site) to track your training runs to gain useful insights into metrics like the eval loss and VRAM usage.


```shell
wandb login
```

Each TrainArgs child class, such as GENTrainArgs has three parameters that are used to control the WandB tracking. 

| Parameter    | Default             | Meaning                              |
|--------------|---------------------|--------------------------------------|
| report_to    | ()                  | Where training results will be saved |
| project_name | "happy-transformer" | Project name within WandB            |
| run_name     | "False"             | If the saved data will be loaded     |

```python
from happytransformer import GENTrainArgs

args = GENTrainArgs(report_to=tuple(["wandb"]), 
                    )
```


