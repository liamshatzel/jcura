import argparse
import os
import time

import torch
import torch_geometric as pyg
from tqdm import tqdm

import dataset
import model
import rollout
import visualize
import wandb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--noise", type=float, default=3e-4)
    parser.add_argument("--eval-interval", type=int, default=100000)
    parser.add_argument("--vis-interval", type=int, default=100000)
    parser.add_argument("--save-interval", type=int, default=100000)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--wandb-sweep", type=bool, default=False)

    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    if torch.cuda.is_available():
        print("Using GPU: " + torch.cuda.get_device_name(0))
    dataset_name = args.data_path.split("/")[-1]
    run_name = dataset_name + f"_{time.strftime('%Y-%m-%d_%H:%M:%S')}"
    config = vars(args)
    wandb.login()

    if args.wandb_sweep: 
        sweep_configuration = {
              "method": "random",
              "metric": {"goal": "minimize", "name": "train_loss"},
              "parameters": {
                #   "batch_size": {"values": [2, 4, 8, 16, 32, 64]},  
                #   "lr_init": {"values": [1e-3, 1e-4, 1e-5]},  
                #   "ntraining_steps": {"min": 500, "max": 1000},
                  "hidden_dim": {"values": [16, 32, 64, 128, 256]},
                #   "mps": {"min": 1, "max": 15},
                #   "conn_radius": {"min": 0.003, "max": 0.03}
              },
          }

        args.hidden_dim = wandb.config.hidden_dim
        sweep_id = wandb.sweep(sweep=sweep_configuration, project="jcura", entity="GAIDG_Lab")
        with wandb.init() as run:
            wandb.agent(sweep_id, function=lambda: train(args), count=10000)
        # wandb.init()


        # args.batch_size = wandb.config.batch_size 
    else: 
        wandb.init(
            project="jcura",
            entity="GAIDG_Lab",
            name=run_name,
            tags=[dataset_name, "train"],
            config=config,
        )
        assert wandb.run is not None
        train(args)

def train(args):
    train_dataset = dataset.OneStepDataset(
        args.data_path, "train", noise_std=args.noise
    )
    valid_dataset = dataset.OneStepDataset(
        args.data_path, "valid", noise_std=args.noise
    )
    train_loader = pyg.loader.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=args.num_workers,
    )
    valid_loader = pyg.loader.DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers,
    )
    rollout_dataset = dataset.RolloutDataset(args.data_path, "valid")

    simulator = model.LearnedSimulator(hidden_size=hidden_dim)
    if torch.cuda.is_available():
        simulator = simulator.cuda()

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(simulator.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=0.1 ** (1 / 5e6)
    )

    total_batch = 0
    for epoch in range(args.epoch):
        simulator.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        total_loss = 0
        batch_count = 0
        for data in progress_bar:
            optimizer.zero_grad()
            if torch.cuda.is_available():
                data = data.cuda()
            pred = simulator(data)
            loss = loss_fn(pred, data.y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            batch_count += 1
            progress_bar.set_postfix(
                {
                    "loss": loss.item(),
                    "avg_loss": total_loss / batch_count,
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )
            wandb.log(
                {
                    "loss": loss.item(),
                    "avg_loss": total_loss / batch_count,
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )

            total_batch += 1
            if args.eval_interval and total_batch % args.eval_interval == 0:
                simulator.eval()
                total_loss = 0
                batch_count = 0
                with torch.no_grad():
                    for data in tqdm(valid_loader):
                        if torch.cuda.is_available():
                            data = data.cuda()
                        pred = simulator(data)
                        loss = loss_fn(pred, data.y)
                        total_loss += loss.item()
                        batch_count += 1
                eval_loss = total_loss / batch_count
                print(f"Eval loss: {eval_loss}")
                wandb.log({"eval_loss": eval_loss})
                simulator.train()

            if args.vis_interval and total_batch % args.vis_interval == 0 and not args.wandb_sweep:
                simulator.eval()
                rollout_data = rollout_dataset[0]
                start = time.time()
                rollout_out = rollout.rollout(
                    simulator, rollout_data, rollout_dataset.metadata, args.noise
                )
                end = time.time()
                rollout_time = end - start
                wandb.log({"rollout_time": rollout_time})
                rollout_out = rollout_out.permute(1, 0, 2)
                anim = visualize.visualize_pair(
                    rollout_data["particle_type"],
                    rollout_out,
                    rollout_data["position"],
                    rollout_dataset.metadata,
                )
                video_path = os.path.join(
                    args.output_path, f"rollout_{total_batch}.mp4"
                )
                anim.save(
                    video_path,
                    writer="ffmpeg",
                    fps=120,
                )
                wandb.log(
                    {
                        "rollout": wandb.Video(
                            video_path,
                            fps=120,
                            format="mp4",
                        )
                    }
                )
                simulator.train()

            if args.save_interval and total_batch % args.save_interval == 0:
                checkpt_path = os.path.join(
                    args.output_path, f"checkpoint_{total_batch}.pt"
                )
                torch.save(
                    {
                        "model": simulator.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "args": vars(args),
                    },
                    checkpt_path,
                )
                artifact = wandb.Artifact(
                    f"checkpoint_{total_batch}", type="model-checkpoint"
                )
                artifact.add_file(checkpt_path)
                wandb.log_artifact(artifact)

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                # save_model_and_train_state(None, device, simulator, args, total_batch, epoch, 
                # optimizer, loss, eval_loss, None, None)

    wandb.finish()


def save_model_and_train_state(rank, device, simulator, flags, step, epoch, optimizer, train_loss, valid_loss, train_loss_hist, valid_loss_hist):
    if rank == 0 or device == torch.device("cpu"):
        model_path = os.path.join(
        flags.output_path, f"model_{step}.pt"
        )
        if device == torch.device("cpu"):
            simulator.save(flags.output_path + 'model_' + str(step) + '.pt')
        else:
            simulator.module.save(flags.output_path + 'model_' + str(step) + '.pt')

        artifact = wandb.Artifact(
            f"model_{step}", type="model")
        artifact.add_file(model_path)
        wandb.log_artifact(artifact)


    #   train_state = dict(optimizer_state=optimizer.state_dict(),
    #                       global_train_state={
    #                         "step": step, 
    #                         "epoch": epoch,
    #                         "train_loss": train_loss,
    #                         "valid_loss": valid_loss
    #                         },
    #                       loss_history={"train": train_loss_hist, "valid": valid_loss_hist}
    #                       )
    #   torch.save(train_state, f'{flags["model_path"]}train_state-{step}.pt')

if __name__ == "__main__":
    main()
