import torch
import torch.nn.functional as F
import random
import numpy as np

def run_decay_experiment(formula_type="old_global", key_type="orthogonal", seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    memory_size = 64
    S = torch.zeros(1, memory_size, memory_size)

    k_target = torch.randn(1, memory_size)
    k_target = F.normalize(k_target, p=2, dim=-1)
    v_target = torch.randn(1, memory_size)
    v_target = F.normalize(v_target, p=2, dim=-1)

    eta = torch.tensor([[1.0]])
    gamma = torch.tensor([[1.0]])

    if formula_type == "old_global":
        S_k = torch.bmm(S, k_target.unsqueeze(-1)).squeeze(-1)
        update_term = torch.bmm((v_target - S_k).unsqueeze(-1), k_target.unsqueeze(1))
        S = gamma.unsqueeze(-1) * S + eta.unsqueeze(-1) * update_term
    elif formula_type == "new_directional":
        S_k = torch.bmm(S, k_target.unsqueeze(-1)).squeeze(-1)
        multiplier = (1.0 - gamma) + (eta * gamma)
        update_term = torch.bmm((eta * v_target - multiplier * S_k).unsqueeze(-1), k_target.unsqueeze(1))
        S = S + update_term

    r_initial = torch.bmm(S, k_target.unsqueeze(-1)).squeeze(-1)

    for i in range(20):
        if key_type == "orthogonal":
            k_new = torch.randn(1, memory_size)
            k_new = k_new - torch.sum(k_new * k_target) * k_target
            k_new = F.normalize(k_new, p=2, dim=-1)
        elif key_type == "similar":
            k_rand = torch.randn(1, memory_size)
            k_new = k_target + 0.5 * k_rand
            k_new = F.normalize(k_new, p=2, dim=-1)
        else: # random
            k_new = torch.randn(1, memory_size)
            k_new = F.normalize(k_new, p=2, dim=-1)

        v_new = torch.randn(1, memory_size)
        v_new = F.normalize(v_new, p=2, dim=-1)

        eta_new = torch.tensor([[1.0]])
        gamma_new = torch.tensor([[0.95]])

        if formula_type == "old_global":
            S_k = torch.bmm(S, k_new.unsqueeze(-1)).squeeze(-1)
            update_term = torch.bmm((v_new - S_k).unsqueeze(-1), k_new.unsqueeze(1))
            S = gamma_new.unsqueeze(-1) * S + eta_new.unsqueeze(-1) * update_term
        elif formula_type == "new_directional":
            S_k = torch.bmm(S, k_new.unsqueeze(-1)).squeeze(-1)
            multiplier = (1.0 - gamma_new) + (eta_new * gamma_new)
            update_term = torch.bmm((eta_new * v_new - multiplier * S_k).unsqueeze(-1), k_new.unsqueeze(1))
            S = S + update_term

    r_final = torch.bmm(S, k_target.unsqueeze(-1)).squeeze(-1)

    sim = F.cosine_similarity(r_initial, r_final).item()
    mag_retention = (torch.norm(r_final) / torch.norm(r_initial)).item()
    return sim, mag_retention

if __name__ == "__main__":
    for seed in [42, 43, 44]:
        print(f"\n================ SEED {seed} ================")
        print("--- Eski Formül (Global Gamma Decay) ---")
        s_r, m_r = run_decay_experiment("old_global", "random", seed)
        print(f"Rastgele Anahtarlar - CosSim: {s_r:.3f}, Büyüklük Koruma: {m_r:.3f}")
        s_o, m_o = run_decay_experiment("old_global", "orthogonal", seed)
        print(f"Ortogonal Anahtarlar - CosSim: {s_o:.3f}, Büyüklük Koruma: {m_o:.3f}")
        s_s, m_s = run_decay_experiment("old_global", "similar", seed)
        print(f"Benzer Anahtarlar  - CosSim: {s_s:.3f}, Büyüklük Koruma: {m_s:.3f}")

        print("\n--- Yeni Formül S(t) + [eta*v - ((1-gamma) + eta*gamma)*S(t)k]k^T ---")
        s_r, m_r = run_decay_experiment("new_directional", "random", seed)
        print(f"Rastgele Anahtarlar - CosSim: {s_r:.3f}, Büyüklük Koruma: {m_r:.3f}")
        s_o, m_o = run_decay_experiment("new_directional", "orthogonal", seed)
        print(f"Ortogonal Anahtarlar - CosSim: {s_o:.3f}, Büyüklük Koruma: {m_o:.3f}")
        s_s, m_s = run_decay_experiment("new_directional", "similar", seed)
        print(f"Benzer Anahtarlar  - CosSim: {s_s:.3f}, Büyüklük Koruma: {m_s:.3f}")
