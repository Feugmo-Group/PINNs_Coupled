import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import pyvista as pv
from matplotlib.colors import LinearSegmentedColormap

#Decorator should go here
class FFN(nn.Module):
    """Fully Connected Feed Forward Neural Net"""
    def __init__(self,cfg,input_dim=2,output_dim=1):
        super(FFN,self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = cfg.nr_layers
        self.layer_size = cfg.layer_size

        self.activation = nn.Tanh()

        # Input layer
        self.input_layer = nn.Linear(input_dim, self.layer_size)

        # Hidden layers
        self.hidden_layers = nn.ModuleList([
            nn.Linear(self.layer_size, self.layer_size) 
            for _ in range(self.num_layers)
        ])

        # Output layer
        self.output_layer = nn.Linear(self.layer_size, output_dim)

    def forward(self, x):
        x_input = x
        x = self.activation(self.input_layer(x))
        
        for i, layer in enumerate(self.hidden_layers):
                x = self.activation(layer(x))
                
        return self.output_layer(x)
    
#Decorator should probably go here
class PNP():
    def __init__(self,cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create networks
        self.potential_net = FFN(cfg.arch.fully_connected, input_dim=2, output_dim=1).to(self.device)
        self.concentration_net = FFN(cfg.arch.fully_connected, input_dim=2, output_dim=1).to(self.device)


        # Physics parameters
        self.D = cfg.pnp.physics.D
        self.z = cfg.pnp.physics.z

        # Optimizer
        params = list(self.potential_net.parameters()) + list(self.concentration_net.parameters())
        self.optimizer = optim.Adam(
            params,
            lr=cfg.optimizer.adam.lr,
            betas=cfg.optimizer.adam.betas,
            eps=cfg.optimizer.adam.eps,
            weight_decay=cfg.optimizer.adam.weight_decay
        )

        # Scheduler/ Same as in PhysicsNEMO for now
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=cfg.scheduler.tf_exponential_lr.decay_steps,
            gamma=cfg.scheduler.tf_exponential_lr.decay_rate
        )

        # Loss weights
        self.poisson_weight = cfg.pnp.weights.poisson_weight
        self.nernst_weight = cfg.pnp.weights.nernst_weight
        self.bc_weight = cfg.pnp.weights.bc_weight

    def compute_gradients(self, x, y):
        """Compute the gradients of potential and concentration."""
        x.requires_grad_(True)
        y.requires_grad_(True)


        # Compute Forward pass
        inputs = torch.cat([x, y], dim=1) #puts all the input values together into one big list
        u_pred = self.potential_net(inputs) #Acts the neural networks on the inputs to compute u and c 
        c_pred = self.concentration_net(inputs)

        # Compute gradients
        u_x = torch.autograd.grad(
            u_pred, x, grad_outputs=torch.ones_like(u_pred), #u_pred is essentially the u function in our network arch, reatain_graph so that we can compute this many times
            create_graph=True, retain_graph=True
        )[0]
        
        u_y = torch.autograd.grad(
            u_pred, y, grad_outputs=torch.ones_like(u_pred),
            create_graph=True, retain_graph=True
        )[0]
        
        c_x = torch.autograd.grad(
            c_pred, x, grad_outputs=torch.ones_like(c_pred),
            create_graph=True, retain_graph=True
        )[0]
        
        c_y = torch.autograd.grad(
            c_pred, y, grad_outputs=torch.ones_like(c_pred),
            create_graph=True, retain_graph=True
        )[0]
        
        # Second derivatives
        u_xx = torch.autograd.grad(
            u_x, x, grad_outputs=torch.ones_like(u_x),
            create_graph=True, retain_graph=True
        )[0]
        
        u_yy = torch.autograd.grad(
            u_y, y, grad_outputs=torch.ones_like(u_y),
            create_graph=True, retain_graph=True
        )[0]
        
        
        # Compute laplacian
        u_laplacian = u_xx + u_yy

        
        return u_pred, c_pred, u_x, u_y, c_x, c_y, u_laplacian
    

    #calculate the pde loss
    def pnp_residuals(self, x, y):
        """Compute the PNP equation residuals."""
        u, c, u_x, u_y, c_x, c_y, u_laplacian = self.compute_gradients(x, y)
        
        # Poisson equation: ∇²u = 0
        poisson_residual = u_laplacian 
        
        # Additional terms for Nernst-Planck
        # Flux J = -D(∇c + zc∇u)
        # Steady state: ∇·J = 0
        flux_x = -self.D * (c_x + self.z * c * u_x)
        flux_y = -self.D * (c_y + self.z * c * u_y)
        
        # Compute divergence of flux
        x.requires_grad_(True)
        y.requires_grad_(True)
        
        flux_x_grad = torch.autograd.grad(
            flux_x, x, grad_outputs=torch.ones_like(flux_x),
            create_graph=True, retain_graph=True
        )[0]
        
        flux_y_grad = torch.autograd.grad(
            flux_y, y, grad_outputs=torch.ones_like(flux_y),
            create_graph=True, retain_graph=True
        )[0]
        
        nernst_residual = flux_x_grad + flux_y_grad
        
        return poisson_residual, nernst_residual, u, c

    def boundary_loss(self):
        """Compute loss for all boundary conditions."""
        # Left boundary (x=0): c=1
        x_left = torch.zeros(self.cfg.batch_size.BC, 1, device=self.device)
        y_left = torch.rand(self.cfg.batch_size.BC, 1, device=self.device) #Nicely gives us random values between (0,1]
        inputs_left = torch.cat([x_left, y_left], dim=1) #nice input list
        c_left = self.concentration_net(inputs_left)
        left_bc_loss = torch.mean((c_left - 1.0)**2) #MSE between predicted c on left and the desired value
        
        # Right boundary (x=2): c=2, u=0
        x_right = torch.ones(self.cfg.batch_size.BC, 1, device=self.device) * 2.0 #Does the same thing as zeros but with ones, 2 scales it up to be at the right boundary
        y_right = torch.rand(self.cfg.batch_size.BC, 1, device=self.device) #This is uniform sampling on the boundaries
        inputs_right = torch.cat([x_right, y_right], dim=1)
        c_right = self.concentration_net(inputs_right)
        u_right = self.potential_net(inputs_right)
        right_bc_loss = torch.mean((c_right - 2.0)**2) + torch.mean(u_right**2) #Combined BC loss on the right boundary for u and c
        
        # Middle boundary (x=1): u=10
        x_mid = torch.zeros(self.cfg.batch_size.BC, 1, device=self.device)
        y_mid = torch.rand(self.cfg.batch_size.BC, 1, device=self.device)
        inputs_mid = torch.cat([x_mid, y_mid], dim=1)
        u_mid = self.potential_net(inputs_mid)
        mid_bc_loss = torch.mean((u_mid - 10.0)**2)
        
        # Top boundary (y=1): dc/dy=0, du/dy=0
        x_top = torch.rand(self.cfg.batch_size.BC, 1, device=self.device) * 2.0 #Random x values between 0 and 2
        y_top = torch.ones(self.cfg.batch_size.BC, 1, device=self.device)
        y_top.requires_grad_(True)
        inputs_top = torch.cat([x_top, y_top], dim=1)
        c_top = self.concentration_net(inputs_top)
        u_top = self.potential_net(inputs_top)
        
        c_top_y = torch.autograd.grad(
            c_top, y_top, grad_outputs=torch.ones_like(c_top), #Diffrentiate c_top w.r.t the input values to which it was calculated
            create_graph=True, retain_graph=True
        )[0]
        
        u_top_y = torch.autograd.grad(
            u_top, y_top, grad_outputs=torch.ones_like(u_top),
            create_graph=True, retain_graph=True
        )[0]
        
        top_bc_loss = torch.mean(c_top_y**2) + torch.mean(u_top_y**2)
        
        # Bottom boundary (y=0): dc/dy=0, du/dy=0
        x_bottom = torch.rand(self.cfg.batch_size.BC, 1, device=self.device) * 2.0
        y_bottom = torch.zeros(self.cfg.batch_size.BC, 1, device=self.device)
        y_bottom.requires_grad_(True)
        inputs_bottom = torch.cat([x_bottom, y_bottom], dim=1)
        c_bottom = self.concentration_net(inputs_bottom)
        u_bottom = self.potential_net(inputs_bottom)
        
        c_bottom_y = torch.autograd.grad(
            c_bottom, y_bottom, grad_outputs=torch.ones_like(c_bottom),
            create_graph=True, retain_graph=True
        )[0]
        
        u_bottom_y = torch.autograd.grad(
            u_bottom, y_bottom, grad_outputs=torch.ones_like(u_bottom),
            create_graph=True, retain_graph=True
        )[0]
        
        bottom_bc_loss = torch.mean(c_bottom_y**2) + torch.mean(u_bottom_y**2)
        
        # Total boundary loss
        total_bc_loss = left_bc_loss + right_bc_loss + mid_bc_loss + top_bc_loss + bottom_bc_loss #Sum losses
        
        return total_bc_loss
    
    def interior_loss(self):
        """Compute loss for PDE residuals in the interior."""
        # Sample interior points
        x = torch.rand(self.cfg.batch_size.interior, 1, device=self.device) * 2.0 #Random x value sampling on the interval (0,2]
        y = torch.rand(self.cfg.batch_size.interior, 1, device=self.device) #Random y value sampling on the interval (0,1]
        
        # Compute residuals
        poisson_residual, nernst_residual, _, _ = self.pnp_residuals(x, y)
        
        # Weighted loss
        poisson_loss = torch.mean(poisson_residual**2) #MSE of the residuals is how we define our loss
        nernst_loss = torch.mean(nernst_residual**2)
        
        interior_loss = self.poisson_weight * poisson_loss + self.nernst_weight * nernst_loss #Weights set to 1 for benchmarking
        
        return interior_loss, poisson_loss, nernst_loss 
    
    def total_loss(self):
        """Compute total loss."""
        int_loss, poisson_loss, nernst_loss = self.interior_loss()
        bc_loss = self.boundary_loss()
        
        return int_loss + self.bc_weight * bc_loss, poisson_loss, nernst_loss, bc_loss #Add weighting to the BC loss and sum all losses
    
    def train_step(self):
        """Perform one training step."""
        self.optimizer.zero_grad()
        loss, poisson_loss, nernst_loss, bc_loss = self.total_loss()
        loss.backward() 
        self.optimizer.step()
        self.scheduler.step()

        return loss.item(), poisson_loss.item(), nernst_loss.item(), bc_loss.item()
    
    def train(self):
        """Train the model."""
        losses = []
        poisson_losses = []
        nernst_losses = []
        bc_losses = []
        
        # Training loop
        for step in range(self.cfg.training.max_steps):
            loss, poisson_loss, nernst_loss, bc_loss = self.train_step()
            losses.append(loss)
            poisson_losses.append(poisson_loss)
            nernst_losses.append(nernst_loss)
            bc_losses.append(bc_loss)
            
            # Print progress
            if step % self.cfg.training.rec_results_freq == 0:
                print(f"Step {step}, Loss: {loss:.6f}, Poisson: {poisson_loss:.6f}, "
                      f"Nernst: {nernst_loss:.6f}, BC: {bc_loss:.6f}")
                
                # Save if specified
                if step % self.cfg.training.save_network_freq == 0 and step > 0:
                    self.save_model(f"outputs/checkpoints/model_step_{step}")
                    
                    # Visualize if needed
                    if step % self.cfg.training.rec_inference_freq == 0:
                        self.visualize(step)
        
        # Final save
        print(f"Step {step}, Loss: {loss:.6f}, Poisson: {poisson_loss:.6f}, "f"Nernst: {nernst_loss:.6f}, BC: {bc_loss:.6f}")
        self.save_model("outputs/checkpoints/model_final")
        self.visualize("final")

        
        return losses, poisson_losses, nernst_losses, bc_losses
    
    def save_model(self, name):
        """Save model state."""
        torch.save({
            'potential_net': self.potential_net.state_dict(),
            'concentration_net': self.concentration_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, f"{name}.pt")

    def visualize(self, step):
        """Visualize the solutions."""
        self.potential_net.eval()
        self.concentration_net.eval()
        
        # Create a grid
        x = np.linspace(0, 2, 100)
        y = np.linspace(0, 1, 50)
        X, Y = np.meshgrid(x, y)
        
        # Convert to tensor
        xy = torch.tensor(np.vstack([X.flatten(), Y.flatten()]).T, dtype=torch.float32).to(self.device) #What is the shape of this tensor? 
        
        # Predictions
        with torch.no_grad():
            u = self.potential_net(xy).cpu().numpy().reshape(Y.shape[0], X.shape[1])
            c = self.concentration_net(xy).cpu().numpy().reshape(Y.shape[0], X.shape[1])

        #Create custom colormaps matching the paper

    
        # Concentration colormap (green to yellow to red)
        colors_c = [(0.0, 0.8, 0.0),    # Green
                    (0.8, 0.8, 0.0),    # Yellow
                    (0.8, 0.0, 0.0)]    # Red
        c_cmap = LinearSegmentedColormap.from_list("concentration_map", colors_c, N=256)
    
        # Potential colormap (red to yellow to cyan to blue)
        colors_u = [(0.0, 0.0, 0.8),    # Blue (low values = 0)
                (0.0, 0.8, 0.8),    # Cyan
                (0.8, 0.8, 0.0),    # Yellow
                (0.8, 0.0, 0.0)]    # Red (high values = 10)
        u_cmap = LinearSegmentedColormap.from_list("potential_map", colors_u, N=256)
    
        plt.figure(figsize=(10, 5))
        # Plot concentration
        plt.subplot(1, 2, 1)
        plt.contourf(X, Y, c, levels=20, cmap=c_cmap)
        plt.colorbar(label='Cocentration (c)')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Concentration - Step {step}')

        # Plot potential
        plt.subplot(1, 2, 2)
        plt.contourf(X, Y, u, levels=20, cmap=u_cmap)
        plt.colorbar(label='Potential (u)')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Potential - Step {step}')
        
        plt.tight_layout()
        plt.savefig(f"outputs/plots/pnp_solution_{step}.png")
        plt.close()
        
        # Export VTK if needed
        self.export_vtk(X, Y, u, c, step)

    def export_vtk(self, X, Y, u, c, step):
        """Export solution to VTK format."""
        # Create grid
        grid = pv.StructuredGrid(X, Y, np.zeros_like(X))
            
        # Add data
        grid.point_data["potential"] = u.flatten()
        grid.point_data["concentration"] = c.flatten()
            
        # Save VTK
        grid.save(f"outputs/vtk/pnp_solution_{step}.vtk")

@hydra.main(config_path="conf", config_name="torch_config",version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    
    # Create model
    model = PNP(cfg)
    
    # Train
    losses, poisson_losses, nernst_losses, bc_losses = model.train()
    
    # Plot losses
    plt.figure(figsize=(10, 6))
    plt.semilogy(losses, label='Total Loss')
    plt.semilogy(poisson_losses, label='Poisson Loss')
    plt.semilogy(nernst_losses, label='Nernst-Planck Loss')
    plt.semilogy(bc_losses, label='Boundary Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig("outputs/plots/training_losses.png")
    plt.close()

if __name__ == "__main__":
    main()