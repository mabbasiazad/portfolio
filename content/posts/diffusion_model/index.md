+++
title = 'Easy-to-Follow Math & Intuition for Diffusion Models'
date = 2023-12-23
author= ["Mehdi Azad"]
summary = "Diffusion models are defined as a Markov chain of diffusion steps that slowly add random noise to data. Then, they learn to reverse the diffusion process to construct desired data samples from the noise. I have tried to process the math underlying these models in a concise and organized manner."
+++

# Introduction

As a brief summary, diffusion models are defined as a Markov chain of diffusion steps that slowly add random noise to data. Then, they learn to reverse the diffusion process to construct desired data samples from the noise. In this set of notes, I have tried to process the math underlying these models in a concise and organized manner.

![overview](./overview.png)

# Forward diffusion

The forward process involves adding Gaussian noise to the latent variable at time $t-1$ to obtain a new latent variable at time $t$. Markov chain of T steps .

suppose that $q(x)$is real distribution on data and $x_0 \sim q(x)$, the diffusion forward process can be formulated as follows: 

$$
q(x_{t}|x_{t-1}) = \mathcal{N}(x_{t}; \mu_{t} = \sqrt{1-\beta_{t}} x_{t-1},\beta_{t}I) 
$$

$$
q(x_{1:T}|x_0) = \prod_{t=1}^{T}q(x_{t}|x_{t-1})
$$

****The reparameterization trick: tractable closed-form sampling at any timestep****

We don’t have to add noise to $x_{0}$ for $t$ time step in order to get $x_{t}$. We can sample $x_{t}$ at any arbitrary time step in a closed loop form using reparametrization trick.

$$
x_t \sim q(x_t|x_{t-1})
$$

$$
\begin{aligned}
x_t &= \sqrt{1 - \beta_{t}}x_{t-1} + \sqrt{\beta_{t}}\epsilon_{t-1} \\\\
&=\sqrt{\alpha_{t}\alpha_{t-1}}x_{t-2} + \sqrt{1-\alpha_{t}\alpha_{t-1}}\epsilon_{t-2} (*) \\\\
&= ...\\\\
&=\sqrt{\bar\alpha_{t}}x_{0} + \sqrt{1-\bar\alpha_{t}}\epsilon_{0}
\end{aligned} 
$$


$$
\alpha_{t} = 1 - \beta_{t}; \quad \bar{\alpha_{t}}= \prod_{s=0}^t \alpha_{t}; \quad \epsilon_{0}, \epsilon_{1}, ... ,\epsilon_{t-1} \sim \mathcal{N}(0,I)
$$

**Note:** Since all timestep have the same Gaussian noise we will only use the symbol $\epsilon$

(*) recall if we merge two independant Guassians $\mathcal{N}(0, \sigma_{1}^2I)$ and  $\mathcal{N}(0, \sigma_{2}^2I)$ the new distribution is $\mathcal{N}(0, (\sigma_{1}^2+\sigma_{2}^2)I)$

So, use the following distribution to sample $x_{t}$

$$
\boxed{
x_{t} \sim q(x_{t}|x_{0}) = \mathcal{N}(x_{t}; \sqrt{\bar\alpha_{t}}x_{0} , (1 - \bar\alpha_{t})I )
}
$$

$$
\boxed{
x_t = \sqrt{\bar\alpha_{t}}x_{0} + \sqrt{1 - \bar\alpha_{t}}\epsilon_{0}
}
$$

## variance schedule

we can afford larger update step when the sample gets noisier so $\beta_{1} > \beta_{2} > ...>\beta_{T}$ or $\bar\alpha_{1}<\bar\alpha_{2}<...<\bar\alpha_{T}$ 

<p align="center">
<img src="./alpa_beta_scheduler.png" width=350 height=250>
</p>

# Reverse Diffusion Process

sampling process = generation process = reverse diffusion 

We cannot easily estimate $q(x_{t-1}|x_{t})$ because it needs to use the entire dataset (use Bays rule to proove this part)

However, the reverse conditional probability is tractable when conditioned on $x_{0}$.

$$
\boxed{q(x_{t-1}|x_{t}, x_{0}) =\mathcal{N}(x_{t-1};\tilde\mu_{t}(x_{t}, x_{0}), \tilde\beta_{t}I)} \\
$$

$$
\boxed{
\tilde\beta_{t}=\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_{t}}\beta_{t}
}
$$

$$
\boxed{
\tilde\mu_{t}(x_{t},x_{0}) = \frac{\sqrt{\alpha_{t}}(1-\bar\alpha_{t-1})}{1-\bar\alpha_{t}}x_{t}+\frac{\sqrt{\alpha_{t-1}}\beta_{t}}{1-\bar\alpha_{t}}x_{0}
}
$$

**Q**: How can I calculate $\tilde\mu$ without having $x_{0}$?

**A**: Use a model to learn $x_0$ => $x_{\theta}(x_{t}, t) = x_{0}$, or directly learn $\mu_{\theta}(x_{t}, t)$ = $\tilde\mu_{t}(x_{t}, x_{0})$.

<br/>


In oder words, we need to learn a model $p_{\theta}$ to approximate this conditional probability in order to run the reverse diffusion process

$$
p_{\theta}(x_{t-1}|x_{t}) \approx q(x_{t-1}|x_{t}, x_{0})
$$

$$
p_{\theta}(x_{t-1}|x_{t})=\mathcal{N}(x_{t-1};\mu_{\theta}(x_{t}, t), \Sigma_{\theta}(x_{t}, t))
$$

<!-- the probability of **reverse trajectory** will be equal to:

$$
p(x_{0:T}) = p(x_{T})\prod_{t=1}^{T}p(x_{t-1}|x_{t})
$$ -->



We would like to train $\mu_{\theta}$ to predict $\tilde\mu_{t}$.

Note: the model is conditioned on the amount of noise via timestep conditioning.


![algorithm_1](./algorithm_1.png)

<aside>
💡 The reverse diffiusion process can be intutively defined as; given a noisy observation $x_{t}$, we first make a prediction corresponding $x_{0}$, then we use it to obtain a sample $x_{t-1}$through the reverse conditional distribution $q(x_{t-1}|x_{t}, x_{0})$

</aside>

<br/>

<aside>
💡 A Diffusion Model can be trained by simply learning a neural network to predict the original natural image $x_{0}$ from an arbitrary noised version $x_{t}$ and its time index t. However, $x_{0}$ has two other equivalent parameterizations, which leads to two further interpretations for a diffusion model.

</aside>

## **Second Interpretation of Revese Process: $\epsilon_{\theta}(x_{t},t) \approx \epsilon_{0}$**


Thanks to the nice property of reparametrization trick we can define $x_{0}$ with respect $\epsilon_{t}=\epsilon_{0}$

 ⇒ the model $\epsilon_{\theta}^{(t)}(x_{t})$ attempts to predict $\epsilon_{0}$ from $x_{t}$ without knowledge of $x_{0}$

$$
\boxed{
\therefore x_{0} = \frac{1}{\sqrt{\bar\alpha_{t}}}(x_{t} - \sqrt{1-\bar\alpha_{t}}\epsilon_{0})
}
$$

putting $x_{0}$ into the eqaution above for $\tilde\mu_{t}(x_{t},x_{0})$ we have; 

$$
\boxed{
\tilde\mu_{t} = \frac{1}{\sqrt{\alpha_{t}}}(x_{t}-\frac{1- \bar\alpha_{t}}{\sqrt{1-\bar\alpha_{t}}}\epsilon_{0})
}
$$

<!-- The aim in reverse process is to learn a network to approximate conditional probability distributions in the reverse diffiusion process. In other words we want to learn $q(x_{t-1}|x_{t}, x_{0})$ -->

Here we would like to train $\mu_{\theta}$ to predict $\tilde\mu_{t} = \frac{1}{\sqrt{\alpha_{t}}}(x_{t}-\frac{1- \bar\alpha_{t}}{\sqrt{1-\bar\alpha_{t}}}\epsilon_{0})$. 

Therefore, we can set our approximate denoising transition mean $\mu_{\theta}(x_{t},t)$ as; 

$$
\mu_{\theta}(x_{t},t) = \frac{1}{\sqrt{\alpha_{t}}}(x_{t}-\frac{1- \bar\alpha_{t}}{\sqrt{1-\bar\alpha_{t}}}\epsilon_{\theta}(x_{t}, t))
$$

$$
\Sigma_{\theta}(x_{t}, t) = \sigma_{t}^2I \quad \sigma_{t}=\tilde\beta_{t}=\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_{t}}\beta_{t}
$$

$\epsilon_{\theta}(x_{t},t)$ is the estimation of $\epsilon_{0}$ based on information available at time $t$.



### Training a diffusion model for the second interpretation

the corresponding optimization problem becomes 

$$
\boxed{\arg  \min_{\theta} D_{KL}(q(x_{t-1}|x_{t}, x_{0})|| p(x_{t-1}|x_{t}))}
$$

$$
= \arg \min_{\theta} D_{KL}(\mathcal{N}(x_{t-1};\tilde\mu_{t}(x_{t}, x_{0}), \tilde\beta_{t}I) ||\mathcal{N}(x_{t-1};\mu_{\theta}(x_{t}, t), \Sigma_{\theta}(x_{t}, t))) 
$$

$$
\begin{aligned}
L_{t} &= E_{x_0, \epsilon}[\frac{1}{2||\Sigma_{\theta}(x_{t},t)||^{2}}||\mu(x_{t}, x_{0})-\mu_{\theta}(x_{t},t)||^{2}] \\\\
&=E_{x_0, \epsilon}[\frac{1}{2||\Sigma_{\theta}||^{2}}||\frac{1}{\sqrt{\alpha_{t}}}(x_{t}-\frac{1-\alpha_{t}}{\sqrt{1-\bar\alpha_{t}}}\epsilon_{0})-\frac{1}{\sqrt{\alpha_{t}}}(x_{t}-\frac{1-\alpha_{t}}{\sqrt{1-\bar\alpha_{t}}}\epsilon_{\theta}(x_{t},t))||^{2}] \\\\
&= E_{x_0, \epsilon}[\frac{(1-\alpha_{t})^2}{2\alpha_{t}(1-\bar\alpha_{t})||\Sigma_{\theta}||^{2}}||\epsilon_{0}-\epsilon_{\theta}(x_{t},t)||^2]\\\\
&= E_{x_0, \epsilon}[\frac{(1-\alpha_{t})^2}{2\alpha_{t}(1-\bar\alpha_{t})||\Sigma_{\theta}||^{2}}||\epsilon_{0}-\epsilon_{\theta}(\sqrt{\bar\alpha_{t}}x_{0} + \sqrt{1 - \bar\alpha_{t}}\epsilon_{0},t)||^2]
\end{aligned} 
$$


**Simplifiction:** Empirically, training a diffusion model works better with a simplified objective that ignores the weighting term

$$
L_{t}^{simple} = E_{t \sim [1, T], x_{0}, \epsilon_{0}}[||\epsilon_{0}-\epsilon_{\theta}(\sqrt{\bar\alpha_{t}}x_{0} + \sqrt{1 - \bar\alpha_{t}}\epsilon_{0},t)||^2]
$$

![algorithm_2](./algorithm_2.png)

<aside>
💡 Here, $\epsilon_{\theta}(x_{t},t)$ is a neural network that learns to predict the source noise $\epsilon_{0} \sim \mathcal{N}(0, I)$ that generates $x_{t}$ from $x_{0}$. We have therefore shown that learning a variational diffusion model by predicting the original image $x_{0}$ (or $\tilde\mu_{t}$ directly) is equivalent to learning to predict the noise.

</aside>

<br/>

**Question:** why don't we simply subtract $\epsilon_{\theta}(x_{t},t)$ from $x_{t}$ and get $x_{0}$? Instead we calculate the noise added at time step $t$ and subtract it from $x_{t}$ and add additional noise term and get $x_{t-1}$.

hint: Simply subtracting the predicted noise would not account for the complexity and uncertainty involved in the denoising process. We have one uncertainty in neural net calculation and another one in adding the additional noise term.

## Third Interpretation: $s_{\theta}(x_{t},t) \approx \nabla_{x_{t}}\log p(x_{t})$ 

To derive the third common interpretation of Diffusion Models, we appeal to Tweedie’s Formula. In English, Tweedie’s Formula states that the true mean of an exponential family distribution, given samples drawn from it, can be estimated by the maximum likelihood estimate of the samples (aka empirical mean) plus some correction term involving the score of the estimate. In the case of just one observed sample, the empirical mean is just the sample itself. 

mathematically for $z \sim \mathcal{N}(\mu, \Sigma)$, Tweedie’s Formula states that; 

$$
\mathbb{E}[\mu|z] = z + \Sigma \nabla_{z}\log p(z)
$$

recall $q(x_{t}|x_{0}) = \mathcal{N}(x_{t}; \sqrt{\bar\alpha_{t}}x_{0} , (1 - \bar\alpha_{t})I ) := p(x)$

$$
\mathbb{E}[\mu_{x_{t}}|x_{t}] = x_{t} + (1- \bar\alpha) \nabla_{x_{t}}\log p(x_{t})
$$

$$
\sqrt{\bar\alpha_{t}}x_{0} = x_{t} + (1- \bar\alpha) \nabla_{x_{t}}\log p(x_{t})
$$

$$
\boxed{
\therefore x_{0} = \frac{x_{t} + (1 - \bar\alpha_{t}) \nabla\log p(x_{t})}{\sqrt{\bar\alpha_{t}}}
}
$$

putting this parametrization of $x_{0}$ in $\tilde\mu(x_{t,}, x_{0})$ formula we will have; 


$$
\tilde\mu(x_{t}, x_{0}) = \frac{1}{\sqrt{\alpha_{t}}}x_{t} + \frac{1-\alpha_{t}}{\sqrt{\alpha_{t}}}\nabla\log p (x_{t})
$$


then we can set our approximate mean as:

$$
\mu_{\theta}(x_{t}, x_{0}) = \frac{1}{\sqrt{\alpha_{t}}}x_{t} + \frac{1-\alpha_{t}}{\sqrt{\alpha_{t}}}s_{\theta}(x_{t}, t)
$$

and the corresponding optimizatoin problem becomes: 

recall $\arg \min_{\theta} D_{KL}(q(x_{t-1}|x_{t}, x_{0})|| p(x_{t-1}|x_{t}))$

$$
\arg \min_{\theta} \frac{1}{2\beta_t^2} \frac{(1-\alpha_{t})^2}{\alpha_{t}}[||  \nabla_{x} \log p(x_{t})- s_{\theta}(x_{t}, t)||_{2}^{2}]
$$

$s_{\theta}(x_{t}, t)$ is called "score". 

According to the equation above, we can train score-based models by minimizing the **Fisher divergence** between the model and the data distributions. Intuitively, the Fisher divergence compares the squared $l_{2}$ distance between the ground-truth data score and the score-based model. 

Directly computing this divergence, however, is infeasible because it requires access to the unknown data score  $\nabla_{x} \log p(\mathbf{x})$. Fortunately, there exists a family of methods called **score matching** that minimize the Fisher divergence without knowledge of the ground-truth data score. 


### Langevin Dynamics

how to draw sample from a distribution by knowing just its score function 

Langevin dynamics accesses $p(x)$ only through  $\nabla_{x} \log p(\mathbf{x})$

$$
x_{i+1} \leftarrow x_{i}+ \epsilon \nabla_{x} \log p(\mathbf{x}) + \sqrt{2\epsilon} z_{i}
$$

where $z_{i}\sim \mathcal{N}(0, I)$ and $\epsilon > 0$ is a fixed step size. $x_{0}$ is initialized from an arbitrary priror distribution $x_{0} \sim \pi(x)$.

### Score-based models learn "Energy gradients"

In the picture, there is an energy potential surface, with arrows indicating the directions to move downhill, learned by the diffuser model. The motion of the dots illustrates Langevin Dynamics, demonstrating how particles move in response to the energy gradients on the surface.

<p align="center">
<img src="./langevin.gif" width=300 height=300>
</p>

<p align="center">
<img src="./energy.png" width=400 height=250>
</p>



<!-- # Implementation

corruption process

In DDPM and many other diffusion model implementations, the model predicts the noise used in the corruption process (before scaling, so **unit variance noise**). In code, it looks something like:

```python
!pip install -q diffusers
noise_scheduler = DDPMScheduler(num_train_timesteps=1000) 
#how large each step should be
from diffusers import DDPMScheduler, UNet2DModel
```

```python
noise = torch.randn_like(xb) # << NB: randn not rand
noisy_x = noise_scheduler.add_noise(x, noise, timesteps)
model_prediction = model(noisy_x, timesteps).sample
loss = mse_loss(model_prediction, noise) # noise as the target
```

how is training loop and how is sample loop : refere to [hugging face notebooks](https://github.com/huggingface/diffusion-models-class).  -->


<!-- # Summary

Diffusion models are a type of generative model that learn to denoise images by iteratively applying a random diffusion process. The model is trained to predict the original image from the noisy output of the diffusion process. The intuition and math behind diffusion models are explained, including the different interpretations of the reverse process, training a diffusion model, and conditioned generation. Guidance methods, including classifier guidance and CLIP guidance, are also discussed. Stable diffusion is introduced as a way to improve the stability of the diffusion process during training. -->

# References

Weng, Lilian. (Jul 2021). What are diffusion models? Lil’Log. [https://lilianweng.github.io/posts/2021-07-11-diffusion-models/](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)

Calvin, Luo. (Aug 2022). Understanding Diffusion Models: A Unified Perspective. arxiv Preprint [arXiv:2208.11970](https://arxiv.org/abs/2208.11970)

Yang, Song, Generative Modeling by Estimating Gradients of the Data Distribution.[https://yang-song.net/blog/2021/score/](https://yang-song.net/blog/2021/score/)

Hugging Face Diffusion Models Course, [https://github.com/huggingface/diffusion-models-class](https://github.com/huggingface/diffusion-models-class)