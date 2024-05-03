"""
This file is used to train LLMs from huggingface. The script take a json configiguration file as an input.
Implementation based on:
- https://github.com/karpathy/llama2.c/blob/master/train.py

To run on a single GPU:
    python train.py config.json
To run with DDP on 2 gpus on 1 node:
    torchrun --standalone --nproc_per_node=2 --nnodes=1 train.py config.json
"""
# Imports ===============================================================================================================
from utils import *

# Main function =========================================================================================================
def main(config):
    # DDP Setup =========================================================================================================
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        seed_offset = ddp_rank
        assert config["gradient_acc_steps"] % ddp_world_size == 0
        config["gradient_acc_steps"] //= ddp_world_size
    else:
        master_process = True
        seed_offset = 0
        ddp_world_size = 1

    if master_process:
        os.makedirs(config["out_dir"], exist_ok=True)
 
    # Load data =========================================================================================================
    if master_process:
        print("\nLoading data...")
    try:
        train_data = pickle.load(open(config["path_to_data"]+"GPT_data_train.pkl","rb"))
        val_data = pickle.load(open(config["path_to_data"]+"GPT_data_validation.pkl","rb"))
        if master_process:
            print(f"Loaded successfully {len(train_data)} samples for training.\n")
            print(f"Loaded successfully {len(val_data)} samples for validation.\n")
    except Exception as e:
        if master_process:
            print(e)
        return

    # Create dataset ====================================================================================================
    if master_process:
        print("Create datasets...\n")

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"],use_fast=False,trust_remote_code=True)
    
    train_dataset = dataset(train_data, tokenizer, config["max_len"])

    val_dataset = dataset(val_data, tokenizer, config["max_len"])
    val_dataloader = DataLoader(val_dataset,
                                    num_workers = config["num_workers"],
                                    batch_size = config["batch_size"])

    if ddp:
        train_sampler = DistributedSampler(train_dataset,
                                            num_replicas = ddp_world_size,
                                            rank = ddp_rank,
                                            seed = config["seed"])    

        train_dataloader = DataLoader(train_dataset,
                                            sampler = train_sampler,
                                            num_workers = config["num_workers"], 
                                            batch_size = config["batch_size"])
    else:
        train_dataloader = DataLoader(train_dataset,
                                            num_workers = config["num_workers"],
                                            batch_size = config["batch_size"])

    # Load model ========================================================================================================
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[config["dtype"]]
    if master_process:
        print("Loading model...")

    if config["quantization"]:
        quant = config["quantization"]
        if quant == "4bit":
            bnb_config = BitsAndBytesConfig(
                                        load_in_4bit=True,
                                        bnb_4bit_use_double_quant=True,
                                        bnb_4bit_compute_dtype=torch.bfloat16)

            model = AutoModelForCausalLM.from_pretrained(config["model_name"], quantization_config=bnb_config)
            model = prepare_model_for_kbit_training(model)            

        elif quant == "8bit":
            bnb_config = BitsAndBytesConfig(
                                        load_in_8bit=True,
                                        bnb_8bit_use_double_quant=True,
                                        bnb_8bit_compute_dtype=torch.bfloat16)

            model = AutoModelForCausalLM.from_pretrained(config["model_name"], quantization_config=bnb_config)
            model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(config["model_name"], torch_dtype=ptdtype,trust_remote_code=True)
        
    lora_config = LoraConfig(
                                r=config["lora_rank"],
                                lora_alpha=config["lora_alpha"],
                                target_modules=config["target_modules"], 
                                bias="none",
                                lora_dropout=0.05,
                                task_type="CAUSAL_LM")   

    model = get_peft_model(model, lora_config)

    # Parameter handling ================================================================================================
    if master_process:
        print("Parameter analyisis:")
        stats = model_parameter_stats(model)
        print(f"Trainable parameters: {stats[0]}")
        print(f"Non-trainable parameters: {stats[1]}")
        print(f"Percentage of trainable parameters: {stats[2]:.2f}%\n") 

    # Training setup ====================================================================================================
    torch.manual_seed(config["seed"] + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = config["device"]
    autocast = (
        nullcontext() if device_type != "cuda"
        else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    )

    model.to(config["device"])
    scaler = torch.cuda.amp.GradScaler(enabled=(config["dtype"] == "float16"))
    max_iters = config["epochs"]*len(train_dataloader)+1
    optimizer = PagedAdamW8bit(params=model.parameters(),
                                    lr = get_lr(1,config,max_iters),
                                    betas = (config["betas"][0],config["betas"][1]),
                                    eps = config["eps"],
                                    weight_decay = config["weight_decay"])

    if config["compile"]:
        if master_process:
            print("Compiling the model...")
        model = torch.compile(model)

    if ddp:
        prefix = "_orig_mod." if config["compile"] else ""
        model._ddp_params_and_buffers_to_ignore = {prefix + "freqs_cis"}
        model = DDP(model, device_ids=[ddp_local_rank])

    if config["wandb_log"] and master_process:
        wandb.init(project=config["wandb_project"], name=config["run_name"], config=config)
        wandb.define_metric("epoch")
        wandb.define_metric("train_step")
        wandb.define_metric("lr", step_metric="train_step")
        wandb.define_metric("effective_batch_loss", step_metric="train_step")
        wandb.define_metric("avg_train_loss", step_metric="epoch")
        wandb.define_metric("val_loss", step_metric="epoch")
        wandb.watch(model, log="all")

    # Train the model ===================================================================================================
    lowest_val_loss = 1.0e9
    for epoch in (pbar := tqdm(range(config['epochs']), leave=False, disable=not master_process)):
        pbar.set_description(f"Epoch {epoch+1}")
        model.train()

        accumulated_loss = 0.0
        effective_batch_loss_accumulator = 0.0

        if ddp:
            train_sampler.set_epoch(epoch)
            raw_model = model.module
        else:
            raw_model = model
        
        batch_pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), leave=False, disable=not master_process)
        
        for i, batch in batch_pbar:
            batch_pbar.set_description(f"Train Step {i}/{len(train_dataloader)}")
            total_steps = epoch * len(train_dataloader) + i
            
            lr = get_lr(total_steps+1,config,max_iters)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            
            if config['wandb_log'] and master_process:
                wandb.log({"train_step": total_steps, "lr": lr})

            if ddp:
                model.require_backward_grad_sync = (i + 1) % config['gradient_acc_steps'] == 0 or (i + 1) == len(train_dataloader)

            ids = batch['ids'].to(config["device"], dtype = torch.long)
            mask = batch['mask'].to(config["device"], dtype = torch.long)
            targets = batch['label'].to(config["device"], dtype = torch.long)

            with autocast:
                outputs = model(input_ids=ids, attention_mask=mask, labels=targets)
                loss = outputs.loss
                loss = loss / config["gradient_acc_steps"]
                effective_batch_loss_accumulator += loss.item()
                accumulated_loss += loss.item()
            
            scaler.scale(loss).backward()

            if (i + 1) % config['gradient_acc_steps'] == 0 or (i + 1) == len(train_dataloader):

                if config['wandb_log'] and master_process:
                    wandb.log({"train_step": total_steps, "effective_batch_loss": effective_batch_loss_accumulator})
                    effective_batch_loss_accumulator = 0.0

                if config['grad_clip'] > 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        if config['wandb_log'] and master_process:
            wandb.log({"epoch": epoch+1, "avg_train_loss": accumulated_loss/len(train_dataloader)})

        # Validation ====================================================================================================
        model.eval()
        accumulated_val_loss = 0.0
    
        batch_val_pbar = tqdm(enumerate(val_dataloader), total=len(val_dataloader), leave=False, disable=not master_process)

        with torch.no_grad():
            for i, batch in batch_val_pbar:
                batch_val_pbar.set_description(f"Validation Step {i}/{len(val_dataloader)}")
                
                ids = batch['ids'].to(config["device"], dtype = torch.long)
                mask = batch['mask'].to(config["device"], dtype = torch.long)
                targets = batch['label'].to(config["device"], dtype = torch.long)

                with autocast:
                    outputs = model(input_ids=ids, attention_mask=mask, labels=targets)
                    loss = outputs.loss
                
                accumulated_val_loss += loss.item()
            
            if master_process:
                avg_val_loss = accumulated_val_loss/len(val_dataloader)
                if config['wandb_log']:
                    wandb.log({"epoch": epoch+1, "val_loss": avg_val_loss})
            
                if avg_val_loss < lowest_val_loss:
                    lowest_val_loss = avg_val_loss
                    raw_model.save_pretrained(f"{config['out_dir']}{config['run_name']}_{epoch+1:02}")

# Handle execution ======================================================================================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <path_to_json_file>")
        sys.exit(1)
    json_config = sys.argv[1]
    try:
        with open(json_config, 'r') as file:
            config = json.load(file)
            main(config)
    except Exception as e:
        print(f"{e}")
        sys.exit(1)
