## Auto-Encoding Variational Bayes

#### Unsupervised learning

**生成模型**：给定训练集，产生与训练集同分布的样本。

$\text{Train data}\sim p_{data}(x),\text{Generating samples}\sim p_{model}(x)$

无监督学习里的一个核心的问题：**密度估计问题**

几种典型的思路：

- 显式的密度估计：显式的定义，并求解分布$p_{model}(x)$
- 隐式的密度估计：学习一个模型$p_{model}(x)$，而无需显式的定义它。

<img src="figures\generative_models.png" width="60%" align=left>

**VAE**：显式概率密度，但方程不可解，用近似方法。 



###### PixelRNN：

利用链式准则将图像$x$的生成概率转变为每个像素生成概率的乘积

**图像**$x$**的似然**：$p(x)=\prod\limits_{i=1}^np(x_i|x_1,\dots,x_{i-1})$，最大化训练数据的似然。

<font color="blue">**需要定义像素的产生顺序。**</font>

<font color="blue">**这个分布很复杂，但是可以使用神经网络来建模。**</font>

可以将像素生成看成一个序列生成的问题，利用**RNN**/**LSTM**的序列描述能力来生成新的像素。

<font color="blue">**缺陷：序列生成，测试速度都非常慢。**</font>

###### PixelCNN

基于已经生成的像素，利用**CNN**来生成新的像素。

可以只用卷积核涉及到的地方参与计算，会加快速度。<font color="blue">**图像的产生过程还是逐像素的序列生成，所以仍然很慢。**</font>

**PixelRNN和PixelCNN**:

- *优点*：
  - 似然函数可以精确的计算
  - 利用似然函数的值可以有效地评估模型的性能
- *缺点*：
  - 序列产生 $\Rightarrow$ 慢

##### （Variational Autoencoders）VAE

###### Auto-Encoding

**自编码器**：无监督的特征学习，其目标是利用无标签数据找到一个有效地低维的特征提取器。

<img src="figures\autoEncoding.png" width="20%" align=left></img>

$z$的维度一般小于$x$的维度（**特征降维**）<font color="red">**希望降维后的特征仅保留数据中有意义的信息。**</font>

1. 早先方法：Linear + nonlinearity (sigmoid)
2. 卷积神经网络流行之前：Deep, fully-connected
3. 卷积神经网络流行后： ReLU CNN

**如何学习**：自编码利用重构损失（解码器）来训练低维的特征表示  

<img src="figures/learning.png" width="30%" align=left></img>

训练后移除解码器（decoder） 

已经训练完毕的编码器可以作为**有监督学习**的初始特征提取模型。

<img src="figures/encoder.png" width="30%" align=left>

保留低维特征$z$，再去进行分类，对于提升网络性能有着极其好的作用。

利用少量有标签数据，训练最终的网络。

微调编码器的时候，编码器已经经过学习。

**但分类性能仍然不如有监督学习，因为生成式会重构特征，而不会去寻找最重要的分类特征**

###### Decoding

在降维后的码空间内随机采样一个编码，可以通过解码器来生成一个新的样本。

######  VAE

使用**VAE**的原因：传统编码器无法学习编码后的空间，也无法做出决策。因此使用变分自编码器。

- 生成$z$分布，并且从$z$分布中采样去进行生成。

<img src="figures/vae.png" width="50%" align=left></img>

编码器生成一个分布（均值/方差），并通过从正态分布采样输出特征。

*原始编码*：$m_1,m_2,m_3/\sigma_1,\sigma_2,\sigma_3$

*带噪声的编码*：$c_1,c_2,c_3$

一维：$z=m_1+\exp(\sigma_1)\cdot e_1$，从方差中采到一个偏移量。

*噪声的方差*（$\sigma_1,\sigma_2,\sigma_3$）是从数据中学习到的。

噪声取指数，保证其是个正数。

- 除重构损失外，需要最小化 $\sum\limits_{i=1}^{3}\left(\exp(\sigma_1)-(1+\sigma_i)+(m_i)^2\right)$
- **目的**: $\sigma\to0\hspace{1.0em}\hat{\sigma}=\exp(\sigma)\to1$，最后一项为$L_2$正则化，希望编码得到的码值不要过于集中，而是稀疏比较好。

###### 推导

*高斯混合模型*：$P(x)=\sum\limits_mP(m)P(x|m)$

**采样**：$m\sim P(m)$ 多项式分布，$m$为整数。

- 采到第$m$个高斯后，$x|m\sim\mathcal N(\mu^m,\Sigma^m)$

Each $x$ you generate is from a mixture.

Distributed representation is better than cluster.

*VAE*：$z\sim\mathcal N(0,I)$。 $z$的每一维表示一个属性，调整每一维的属性可以调整对应生成图片的属性。

**采样**：$x|z\sim\mathcal N(\mu(z),\sigma(z))$

将$z$输入到神经网络内，来学习$\mu(z)$和$\sigma(z)$

 变分自编码器采样：$P(x)=\int\limits_zP(z)P(x|z)dz$

<font color="blue">**虽然$z$采样自$\mathcal N(0,1)$，但是$P(x)$可以具有很复杂的形式。**</font>

**极大似然估计**
$P(z)$正态分布，$x|z\sim\mathcal N(\mu(z),\sigma(z))$，其中$\mu(z)$和$\sigma(z)$为待估计的参数。

$P(x)=\int\limits_zP(z)P(x|z)dz$，不同的$z$码对应不同的高斯。

$L=\sum\limits_{x}\log P(x)$

**VAE为近似方法的原因**：由于有隐变量$z$，无法最大化似然，无法通过穷举所有的$z$进行积分，<font color="red">$p(x|z)$**无法写出来，而是通过神经网络学习得到的**</font>

但并不是所有的$z$都能产生高斯到图像，所以我们只考虑数据可以产生的$z$，利用数据产生的$z$进行学习。

我们需要另一个分布$q(z|x)$：$z|x\sim\mathcal N(\mu'(x),\sigma'(x))$.

通过编码器得到$\mu'(x),\sigma'(x)$，通过解码器得到$\mu(z),\sigma(z)$

<img src="figures/encodeZ.png" width="20%" align=left>

<img src="figures/encodeX.png" width="20%" align=left>

**考虑**：$\log P(x)=\int\limits_zq(z|x)\log P(x)dz$

- 恒等式引入$q(z|x)$，等式本身与$\log p(x)$无关，而$\int\limits_zq(z|x)dz=1$，此时的$q(z|x)$可以是任何分布

**推导**：
$$
\begin{aligned}
	\log P(x)&=\int\limits_zq(z|x)\log P(x)dz\\
	&=\int\limits_zq(z|x)\log\left(\frac{P(z,x)}{P(z|x)}\right)dz&\text{贝叶斯公式}\\
	&=\int\limits_zq(z|x)\log\left(\frac{p(z,x)q(z|x)}{q(z|x)p(z|x)}\right)&\log\text{公式内引入}q(z|x)\\
	&=\int\limits_zq(z|x)\log\left(\frac{p(z,x)}{q(z|x)}\right)dz+\int\limits_{z}q(z|x)\log\left(\frac{q(z|x)}{p(z|x)}\right)dz&分解\\
	&=\int\limits_zq(z|x)\log\left(\frac{p(z,x)}{q(z|x)}\right)dz+KL(q(z|x)||p(z|x))\\
	&第二项KL散度一定是大于0的，衡量两个分布之间的距离，但是由于无监督，本项并不参与计算\\
	&\geq\int\limits_zq(z|x)\log\left(\frac{p(x|z)p(z)}{q(z|x)}\right)dz
\end{aligned}
$$
其中，$q(z|x)\log\left(\frac{p(x|z)p(z)}{q(z|x)}\right)$是$\log p(x)$的下界函数$L_b$，因此优化此项，也是优化最大似然。 
$$
\begin{aligned}
L_b&=\int\limits_zq(z|x)\log\left(\frac{P(z,x)}{q(z|x)}\right)\\
&=\int\limits_zq(z|x)\log\left(\frac{p(x|z)p(z)}{q(z|x)}\right)&分解\\
&=\int\limits_zq(z|x)\log\left(\frac{p(z)}{q(z|x)}\right)+\int\limits_zq(z|x)\log p(x|z)dz\\
&=-KL(q(z|x)||p(z))+\int\limits_zq(z|x)\log P(x|z)dz
\end{aligned}
$$
**与神经网络之间的联系**：

分解$L_b=A_1+A_2$:

1. 最小化 
   $$
   A_1=KL(q(z|x)||p(z))
   $$
   等价于最小化
   $$
   \sum\limits_{i=1}^3\left(\exp(\sigma_i)-(1+\sigma_i)+(m_i)^2\right)
   $$
   

2. 最大化(离散的$z$下，取得$x$的期望)
   $$
   A_2=\int\limits_zq(z|x)\log P(x|z)dz=E_{q(z|x)}[\log P(x|z)]
   $$

3. 

<img src="figures/pipe.png" width="50%" align=left></img>

###### 总结

基于典型自编码器拓展成的概率框架$\Rightarrow$可以产生新的样本

定义一个难以计算的密度函数$\Rightarrow$通过推导来优化一个下边界

**优点**：

- 作为生成模型里的一种典型的方法
- 可以计算$q(z|x)$，这个特征表示可以用在其他的许多任务中。

**缺点**：

- 最大化似然函数的下边界能够有效地工作，但是模型本身并不像PixelRNN或者PixelCNN那样好评估。
- 与最新的技术(GANs)相比，产生的样本较模糊，质量较低。

### Intuition

(High dimensional) **variable** $x$: generated from conditional distribution $p_{\theta^*}(x|z)$

**Unobserved continuous random variable** $z$: generated from prior distribution $p_{\theta^*}(z)$

<img src="figures/zx.png" width="15%" align=left>

<font color="orange"># 假设$z$是满足一定分布的，那么也会有从参数$\theta$到$z$的箭头。星号$\theta^*$代表Groundtruth，之后提到的$\theta$均为Decoder的参数。$\phi$为Encoder的参数</font>

**Intractability**:

-  $p_\theta(z|x)=p_\theta(x|z)p_\theta(z)/p_\theta(x)$

- $p_\theta(x)=\int\limits_zp_\theta(z)p_\theta(x|z)dz$

**Approximation:**

- $p_\theta(z|x)\overset{\sim}{=}q_\phi(z|x)$ <font color="orange"># 推导已经写在前面的部分上了</font>

$\log(p_\theta(x))=KL(q_\phi(z|x)||p_\theta(z|x))+L(\theta,\phi;x)$ <font color="red">	**non-negative + Variation Lower Bound**</font>

**Maximize the lower bound**：

$L(\theta,\phi;x)=E_{q_{\phi}(z|x)}[\log(p_\theta(x|z))]-D_{KL}(q_\phi(z|x)||p_\theta(z))$	<font color="red">**Reconstruction Loss + Regularization Loss**</font>

<font color="orange"># 前一项希望从$z$推导出的$x$和我们实际的$x$尽量接近，第二项希望两个分布尽量接近</font>

**Gradient Estimation**: Naive Monte Carlo gradient estimator.有着方差比较大的导数。

**Generic Stochastic Gradient Variational Bayes (SGVB) estimator**

KL散度部分的Loss：
$$
-\frac{1}{2}\sum\limits_{j=1}^J\left(1+\log((\sigma_j)^2)-(\mu_j)^2-(\sigma_j)^2\right)
$$
期望部分的Loss:
$$
f^*=\arg\max\limits_{f\in F}\mathbb{E}_{z\sim{q_x^*}}\left(-\frac{||x-f(z)||^2}{2c}\right)\hspace{2.0em} \text{MSE}
$$
<img src="figures\SGVB.png" width="60%" align=left></img>

## VQ-VAE: Neural Discrete Representing Learning

VQ-VAE编码得到的编码向量是离散化的，<font color="red">**编码向量的每个元素都是一个整数。**</font> Quantised(量子化)

**PixelCNN**：

$p(x)=p(x_1)p(x_2|x_1)\dots p(x_{3n^2}|x_1,x_2\dots,x_{3n^2-1})$，其中每一个概率都是一个$256$分类问题

**自回归模型**：如何设计递归顺序/如何加速采样过程。

但生成像素仍然很耗时，序列很长，不管是RNN还是CNN都无法很好的捕捉长依赖。

且原始的自回归割裂了类别之间的联系，因为连续像素之间的差别是很小的，如果使用交叉熵来预测会带来很大的损失（像素值$100$或是$99$本质上的区别并不大）

<font color="red">**VQ-VAE：先降维，再对编码向量使用PixelCNN建模**</font>

**问题**：

- 常见的降维手段，如自编码器，生成的编码都是连续性变量，无法直接生成离散的变量。
- 同时，生成离散型变量还意味着存在梯度消失的问题。
- 且降维，重构要保证重构之后的图像不失真。

###### 最邻近重构

图片$x\in\R^{n\times n\times 3}$被传入$\text{encoder}$中，得到连续的编码向量$z$
$$
z=\text{encoder}(x),z\in\R^d
$$
同时，VQ-VAE维护一个$\text{Embedding}$层，也可以成为编码表，记为
$$
E=[e_1,e_2\dots,e_K],e_i\in\R^d
$$
随后，通过最邻近搜索，将$z$映射为这$K$个向量之一。
$$
z\rightarrow e_k,\hspace{2.0em}k=\arg\min\limits_j||z-e_j||_2
$$
将$z$对应的编码表向量记作$z_q$，最后传入一个$\text{decoder}$，希望重构原图$\hat{x}=\text{decoder}(z_q)$

由于$z_q$是编码表$E$中的向量之一，所以它实际等价于$1,2,\dots,K$这$K$个整数之一。因为整个流程相当于把图片编码为了一个整数

但若只编码为一个向量，重构的时候可能会失真，泛化性难以得到保障，因此实际编码的时候使用多层卷积，将$x$编码为$m\times m$个大小为$d$的向量。
$$
z=\left(
	\begin{array}{cccc}
		z_{11}&z_{12}&\dots&z_{1m}\\
		z_{21}&z_{22}&\dots&z_{2m}\\
		\vdots&\vdots&\ddots&\vdots\\
		z_{m1}&z_{m2}&\dots&z_{mm}
	\end{array}
\right),z\in\R^{m\times m\times d}
$$
此时，在保留原来位置结构的基础上，每个向量都用前述方法映射为编码表中的一个，可以得到一个同样的大小的$z_q$，并且进一步用它来重构。

$z_q\in\R^{m\times m}$

###### 自行设计梯度（Straight-Through Estimator）

普通自编码器的训练Loss: $||x-\text{decoder}(z)||_2^2$

VQ-VAE: $||x-\text{decoder}(z_q)||_2^2$

**问题**：$z_q$的建构过程中包含了$\arg\min$，这个操作没有梯度，无法反向更新$\text{encoder}$

<font color="red">**Straight-Through Estimator**</font>：

前向传播的时候可以用想要的变量（哪怕不可导），反向传播的时候利用设计好的梯度

因此，目标函数：$||x-\text{decoder}(z+sg[z_q-z])||_2^2$

反向传播的时候根据$\text{decoder}(z)$来反向传播

###### 维护编码表

期望$z_q$与$z$本身很接近，因此将$||z-z_q||_2^2$添加到损失中
$$
||x-\text{decoder}(z+sg[z_q-z]||_2^2+\beta||z-z_q||_2^2
$$
更进一步，应该让$z_q$靠近$z$而不是$z$靠近$z_q$，因为$z$要保证重构的效果，因此，将上式等价分解为
$$
||sg[z]-z_q||_2^2+||z-sg[z_q]||_2^2
$$
并可以分别调整这两项的比例，其中$\gamma<\beta$，原文中$\gamma=0.25\beta$
$$
||x-\text{decoder}(z+sg[z_q-z])||_2^2+\beta||sg[z]-z_q||_2^2+\gamma||z-sg[z_q]||_2^2
$$

###### 拟合编码分布

将图片编码为$m\times m$的整数矩阵后，其一定程度上保留了原来输入的图片的位置信息，可以使用自回归模型来对其进行拟合。得到编码分布，可以随机生成一个新的编码矩阵，然后通过编码表$E$来映射为$3$维的实数矩阵$z_q$（行\*列\*编码的宽度），最终通过$\text{decoder}$得到一张图片