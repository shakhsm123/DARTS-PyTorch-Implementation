import torch
import torch.nn.functional as F


class Architect_first_order:
    def __init__(self, model, arch_lr, arch_weight_decay):
        self.model = model
        self.optimizer = torch.optim.Adam(
            self.model.arch_parameters(), lr=arch_lr, weight_decay=arch_weight_decay
        )

    def step(self, x_val, y_val):
        self.optimizer.zero_grad()
        logits = self.model(x_val)
        loss = F.cross_entropy(logits, y_val)
        loss.backward()
        self.optimizer.step()


class Architect_second_order(Architect_first_order):
    def step(self, x_val, y_val, x_train, y_train, xi):
        self.optimizer.zero_grad()
        arch_params = set(self.model.arch_parameters())
        w = [p for p in self.model.parameters() if p not in arch_params]

        logits_train = self.model(x_train)
        train_loss = F.cross_entropy(logits_train, y_train)
        dw = torch.autograd.grad(train_loss, w)
        w_prime = [p - xi * d for p, d in zip(w, dw)]

        original_weights = [p.data.clone() for p in self.model.parameters()]

        for p, p_prime in zip(w, w_prime):
            p.data.copy_(p_prime)

        logits_val = self.model(x_val)
        val_loss = F.cross_entropy(logits_val, y_val)

        d_w_prime = torch.autograd.grad(val_loss, w, retain_graph=True)
        d_alpha = torch.autograd.grad(val_loss, self.model.arch_parameters())

        for p, p_original in zip(self.model.parameters(), original_weights):
            p.data.copy_(p_original)

        epsilon = 0.01 / torch.cat([g.flatten() for g in d_w_prime]).norm()

        w_plus = [p + epsilon * g for p, g in zip(w, d_w_prime)]
        w_minus = [p - epsilon * g for p, g in zip(w, d_w_prime)]

        for p, p_w_plus in zip(w, w_plus):
            p.data.copy_(p_w_plus)
        logits_plus = self.model(x_train)
        plus_loss = F.cross_entropy(logits_plus, y_train)
        grad_alpha_plus = torch.autograd.grad(plus_loss, self.model.arch_parameters())

        for p, p_original in zip(self.model.parameters(), original_weights):
            p.data.copy_(p_original)

        for p, p_w_minus in zip(w, w_minus):
            p.data.copy_(p_w_minus)
        logits_minus = self.model(x_train)
        minus_loss = F.cross_entropy(logits_minus, y_train)
        grad_alpha_minus = torch.autograd.grad(minus_loss, self.model.arch_parameters())

        for p, p_original in zip(self.model.parameters(), original_weights):
            p.data.copy_(p_original)

        correction_term = [
            (g_plus - g_minus) / (2 * epsilon)
            for g_plus, g_minus in zip(grad_alpha_plus, grad_alpha_minus)
        ]
        final_grad = [dv - xi * c for dv, c in zip(d_alpha, correction_term)]

        for param, grad in zip(self.model.arch_parameters(), final_grad):
            param.grad = grad
        self.optimizer.step()
