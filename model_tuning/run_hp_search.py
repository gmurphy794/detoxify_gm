"""Run a hyperparameter search on a RoBERTa model fine-tuned on BoolQ.

Example usage:
    python run_hyperparameter_search.py BoolQ/
"""
import argparse
import src.data_loaders as module_data
from torch.utils.data import DataLoader
import finetuning_utils
from ray import tune
from ray.tune.suggest.bayesopt import BayesOptSearch
import json


from transformers import TrainingArguments, Trainer


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run a hyperparameter search for finetuning a DeBERTa model on the Jigsaw dataset."
    )
        
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )   

    parser.add_argument(
        "-p",
        "--checkpoint",
        default=None,
        type=str,
        help="path to model checkpoint",
    )

    args = parser.parse_args()
    config = json.load(open(args.config))
    checkpoint_path = args.checkpoint

    # Since the labels for the test set have not been released, we will use half of the
    # validation dataset as our test dataset for the purposes of this assignment.
    # train_df = pd.read_json(f"{args.data_dir}/train.jsonl", lines=True, orient="records")
    # val_df, test_df = train_test_split(
    #     pd.read_json(f"{args.data_dir}/val.jsonl", lines=True, orient="records"),
    #     test_size=0.5,
    # )

    # tokenizer = tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base", cache_dir="../.cache")
    # train_data = boolq.BoolQDataset(train_df, tokenizer)
    # val_data = boolq.BoolQDataset(val_df, tokenizer)
    # test_data = boolq.BoolQDataset(test_df, tokenizer)

    # data
    def get_instance(module, name, config, *args, **kwargs):
        return getattr(module, config[name]["type"])(*args, **config[name]["args"], **kwargs)

    dataset = get_instance(module_data, "dataset", config)
    val_dataset = get_instance(module_data, "dataset", config, train=False)

    data_loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )

    valid_data_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        num_workers=args.num_workers,
        shuffle=False,
    )

    ## TODO: Initialize a transformers.TrainingArguments object here for use in
    ## training and tuning the model. Consult the assignment handout for some
    ## sample hyperparameter values.
    training_args = TrainingArguments(output_dir = '/scratch/${USER}/', \
                per_gpu_train_batch_size=10,
                num_train_epochs=3)


    hp_space = {'learning_rate':tune.uniform(1e-05, 5e-05)}


    ## TODO: Initialize a transformers.Trainer object and run a Bayesian
    ## hyperparameter search for at least 5 trials (but not too many) on the 
    ## learning rate. Hint: use the model_init() and
    ## compute_metrics() methods from finetuning_utils.py as arguments to
    ## Trainer(). Use the hp_space parameter in hyperparameter_search() to specify
    ## your hyperparameter search space. (Note that this parameter takes a function
    ## as its value.)
    ## Also print out the run ID, objective value,
    ## and hyperparameters of your best run.


    trainer = Trainer(
        args=training_args,
        gpus=args.n_gpu,
        tokenizer=tokenizer,
        accumulate_grad_batches=config["accumulate_grad_batches"],
        train_dataset=data_loader,
        eval_dataset=valid_data_loader,
        deterministic=False,
        model_init=finetuning_utils.model_init(config, checkpoint_path),
        compute_metrics=finetuning_utils.compute_metrics,
    )

    trainer = pl.Trainer(
        gpus=args.n_gpu,
        max_epochs=args.n_epochs,
        accumulate_grad_batches=config["accumulate_grad_batches"],
        resume_from_checkpoint=args.resume,
        default_root_dir="saved/" + config["name"],
        deterministic=False,
        model_init=finetuning_utils.model_init,
        compute_metrics=finetuning_utils.compute_metrics,
    )

    best_trial = trainer.hyperparameter_search(
        hp_space=lambda _:hp_space,
        compute_objective=lambda x: x['eval_loss'],
        n_trials = 5,
        direction="minimize",
        backend="ray",
        local_dir="./ray_results/",
        search_alg=BayesOptSearch(hp_space, metric="eval_loss", mode="min"))

    print(best_trial)