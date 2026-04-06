def main(args):

    import sys
    import os
    # Ensure local package (GMOCAT-modif) is prioritized for imports
    local_pkg = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if local_pkg not in sys.path:
        sys.path.insert(0, local_pkg)

    import torch
    import numpy as np
    import logging
    import glob
    import matplotlib.pyplot as plt
    from launch_gcat import construct_local_map
    from util import get_objects, set_global_seeds
    import envs as all_envs
    import agents as all_agents
    import function as all_FA
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from analyze_results import parse_log

    # Cek device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Menggunakan device: {device}")
    args.device = device

    # Set seed
    set_global_seeds(args.seed)
    if args.device == 'cuda':
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_no)

    # Setup Logger
    logger = logging.getLogger(f'{args.FA}')
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        import datetime
        os.makedirs(f'baseline_log/{args.data_name}', exist_ok=True)
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        log_file = f'baseline_log/{args.data_name}/GCAT_{args.data_name}_{args.CDM}_{timestamp}.txt'
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        print(f"Log disimpan ke: {log_file}")

    # Setup Environment
    print("Setup Environment...")
    envs = get_objects(all_envs)
    env = envs[args.environment](args)

    # Update args dengan info dari env
    args.user_num = env.user_num
    args.item_num = env.item_num
    args.know_num = env.know_num

    print(f"Environment siap. User: {args.user_num}, Item: {args.item_num}, Concept: {args.know_num}")

    # Load Graph
    print(f"Loading Graph Data untuk {args.data_name}...")
    # Pre-scan graph files to determine if graph indexing requires larger node counts
    # Try local graph_data first (module dir), then fallback to relative path.
    graph_dir_candidates = [
        os.path.join(os.path.dirname(__file__), 'graph_data', args.data_name),
        os.path.join(os.getcwd(), 'graph_data', args.data_name),
        os.path.join(os.path.dirname(__file__), '..', 'graph_data', args.data_name),
    ]
    required_total = None
    for gd in graph_dir_candidates:
        if os.path.isdir(gd):
            max_idx = -1
            for fname in ['K_Directed.txt', 'k_from_e.txt', 'e_from_k.txt']:
                fpath = os.path.join(gd, fname)
                if os.path.exists(fpath):
                    with open(fpath, 'r') as f:
                        for line in f:
                            parts = line.strip().split('\t')
                            if len(parts) >= 2:
                                a, b = int(parts[0]), int(parts[1])
                                if a > max_idx: max_idx = a
                                if b > max_idx: max_idx = b
            if max_idx >= 0:
                required_total = max_idx + 1
                break

    if required_total is not None:
        if required_total > (args.item_num + args.know_num):
            diff = required_total - (args.item_num + args.know_num)
            args.know_num += diff
            print(f"Adjusted args.know_num by +{diff} to accommodate graph indexing: {args.know_num}")

    local_map = construct_local_map(args, path=f'graph_data/{args.data_name}/')

    # Setup Model (Function Approximation)
    print("Membangun Model GCAT...")
    nets = get_objects(all_FA)
    fa = nets[args.FA].create_model(args, local_map)

    # Setup Agent
    print("Inisialisasi Agen...")
    agents = get_objects(all_agents)
    agent = agents[args.agent](env, fa, args)
    print("Sistem siap dilatih.")

    # Jalankan Training
    print("Memulai Training...")
    agent.train()
    print("Training Selesai.")

    # Analisis Hasil & Visualisasi
    log_dir = f'baseline_log/{args.data_name}'
    if os.path.exists(log_dir):
        list_of_files = glob.glob(f'{log_dir}/*.txt')
        if list_of_files:
            latest_file = max(list_of_files, key=os.path.getctime)
            print(f"Menganalisis log: {latest_file}")
            data = parse_log(latest_file)
            if data:
                plt.figure(figsize=(10, 6))
                plt.plot(range(1, len(data) + 1), data, marker='o', label='Concept Coverage')
                plt.title(f'Concept Coverage Curve ({args.data_name})')
                plt.xlabel('Step')
                plt.ylabel('Coverage Ratio')
                plt.grid(True)
                plt.legend()
                plt.show()
                print(f"Final Coverage: {data[-1]*100:.2f}%")
            else:
                print("Data coverage tidak ditemukan di log (mungkin format berbeda atau training belum selesai).")
        else:
            print("Belum ada file log.")
    else:
        print("Direktori log tidak ditemukan.")



# Definisi Argumen (mengadaptasi dari notebook)
class Args:
    def __init__(self):
        # Konfigurasi Dasar
        self.seed = 42
        self.environment = "GCATEnv"
        self.data_path = "./data/"
        self.data_name = "dbekt22" # Target dataset
        self.agent = "GCATAgent"
        self.FA = "GCAT"
        self.CDM = "NCD"
        # Pengaturan Testing
        self.T = 20 # Panjang tes (steps)
        self.ST = [1, 5, 10, 20] # Step evaluasi
        self.student_ids = [0] # Dummy
        self.target_concepts = [0] # Dummy
        # Hyperparameters Training
        self.gpu_no = "0"
        self.device = None # Akan di-set di main
        self.learning_rate = 0.001 #ini dicari yg tepat : 1e-4 (0.0001) 2e-5 (0.00002) 3e-2 (0.002)
        self.training_epoch = 50 # Set ke angka lebih tinggi (misal 20) untuk hasil maksimal
        self.train_bs = 32      # Kurangi jika memory error (misal 32)
        self.test_bs = 1024     
        self.batch = 32 #dikecilin jd 8,16,32
        # Parameter Model & RL
        self.cdm_lr = 0.01
        self.cdm_epoch = 10 # Epoch untuk update CDM per step
        self.cdm_bs = 128
        self.gamma = 0.9
        self.latent_factor = 256
        self.n_block = 2
        self.graph_block = 2
        self.n_head = 1
        self.dropout_rate = 0.1
        self.policy_epoch = 4
        self.morl_weights = [1, 1, 1] # Bobot [Accuracy, Diversity, Novelty] - High Coverage Priority
        self.emb_dim = 64 # menyesuaikan dengan jumlah data
        self.use_graph = True
        self.use_attention = True
        self.store_action = False
        # Akan diisi setelah env setup
        self.user_num = None
        self.item_num = None
        self.know_num = None
    def __str__(self):
        return str(self.__dict__)


if __name__ == "__main__":
    args = Args()
    print("Konfigurasi dimuat.")
    main(args)
