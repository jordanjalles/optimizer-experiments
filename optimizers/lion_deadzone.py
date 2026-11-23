import torch

#lion
class Lion(torch.optim.Optimizer):
    def __init__(
        self, 
        params, 
        lr = 1e-4, 
        beta1 = 0.9, 
        beta2 = 0.99, 
        decay = 0.0,
        deadzone = 0.0
        ):
        defaults = {
            "lr": lr,
            "beta1": beta1,
            "beta2": beta2,
            "decay": decay,
            "deadzone": deadzone
        }

        params = list(params)
        super().__init__(params, defaults)
        
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.decay = decay
        self.deadzone = deadzone
        self.active_fraction = None
        

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()



        active_count = 0
        total_count = 0
        for group in self.param_groups:

            lr = group["lr"]
            b1 = group["beta1"]
            b2 = group["beta2"]
            decay = group["decay"]
            deadzone = group["deadzone"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]

                if 'momentum' not in state:
                    state['momentum'] = torch.zeros_like(p)

                m = state['momentum']

                grad = p.grad


                #blend current gradient with stored momentum via beta
                update = b1*m + (1-b1)*p.grad
                
                #decoupled weight decay
                p.mul_(1- decay*lr)

                #update each weight by only the directions -1, 0, or +1
                active = update.abs() > deadzone
                direction = update.sign()
                direction.mul_(active)

                p.add_( direction, alpha= -lr)

                # Accumulate global stats
                active_count = active_count + active.sum()
                total_count += active.numel()

                #update the stored momentum for the next step
                m.mul_(b2).add_( p.grad, alpha = (1-b2))            

        self.active_fraction = active_count / total_count

        return loss 