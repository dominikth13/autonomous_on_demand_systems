import math
import torch
import torch.nn as nn

from program.program_params import ProgramParams

class TemporalDifferenceLoss(nn.Module):
    def __init__(self):
        super(TemporalDifferenceLoss, self).__init__()

    def forward(self, main_net, trajectories_and_state_values):
        loss = 0
        for x in trajectories_and_state_values:
            start_state_value = x[1]
            end_state_value = x[2]
            start_t = x[0]["current_time"]
            end_t = x[0]["target_time"]
            duration = end_t - start_t if end_t > start_t else 86400 - start_t + end_t
            reward = x[0]["reward"]
            loss += (
                (
                    (reward * (ProgramParams.DISCOUNT_FACTOR(duration) - 1))
                    / (duration * (ProgramParams.DISCOUNT_FACTOR(1) - 1))
                )
                + ProgramParams.DISCOUNT_FACTOR(duration) * end_state_value
                - start_state_value
            ) ** 2
        
        # TODO find out what this Lipschitz constant is
        return loss + math.exp(-4) * TemporalDifferenceLoss.lipschitz_regularizer(main_net, trajectories_and_state_values)
    
    def lipschitz_regularizer(model, trajectories_and_state_values):
        reg_terms = []
        for x in trajectories_and_state_values:
            with torch.enable_grad():
                # Berechnen des Outputs für einen Eingabesatz
                input = torch.Tensor([x[0]["current_zone"]])
                input.requires_grad = True
                start_state_value = model(input)
                # Berechnen des Gradienten der Ausgabe in Bezug auf die Eingabe
                gradients = torch.autograd.grad(outputs=start_state_value, inputs=input, 
                                                create_graph=True,allow_unused=True, retain_graph=True, only_inputs=True)[0]

                # Berechnen und speichern der Norm des Gradienten
                grad_norm = gradients.norm()
                reg_terms.append(grad_norm)

        # Mitteln der Regularisierungsterme über den gesamten Batch
        avg_reg_term = torch.mean(torch.stack(reg_terms))
        
        # Lipschitz-Regularisierungsterm
        return avg_reg_term