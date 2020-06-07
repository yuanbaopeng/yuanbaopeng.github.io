---
title: xLearn源码分析之FM的CalcScore实现
date: 2019-01-24 12:15:21
tags: 
 - C++
 - xLearn
categories: xLearn
mathjax: true
---
{% include mathjax.html %}
## 写在前面

xLearn是由Chao Ma实现的一个高效的机器学习算法库，这里附上github地址：

[https://github.com/aksnzhy/xlearn](https://github.com/aksnzhy/xlearn)

FM是机器学习中一个在CTR领域中表现突出的模型，最早由Konstanz大学Steffen Rendle（现任职于Google）于2010年最早提出。

## FM模型

FM的模型方程为:

$$y(\mathbf{x}) = w_0+ \sum_{i=1}^n w_i x_i + \sum_{i=1}^n \sum_{j=i+1}^n \langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j​$$

直观上看，FM的复杂度是 $O(kn^2)$，但是，FM的二次项可以化简，其复杂度可以优化到 $O(kn)$。论文中简化如下式：

$$\sum_{i=1}^n \sum_{j=i+1}^n \langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j = \frac{1}{2} \sum_{f=1}^k \left(\left( \sum_{i=1}^n v_{i, f} x_i \right)^2 - \sum_{i=1}^n v_{i, f}^2 x_i^2 \right)​$$

这里记录一下具体推导过程：

$$\sum_{i=1}^n \sum_{j=i+1}^n \langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j ​$$

$$= \frac{1}{2}\left(\sum_{i=1}^n \sum_{j=1}^n \langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j - \sum_{i=1}^n \langle \mathbf{v}_i, \mathbf{v}_i \rangle x_i x_i \right)​$$

$$= \frac{1}{2}\left(\sum_{i=1}^n \sum_{j=1}^n \sum_{f=1}^k v_{if} v_{jf} x_i x_j - \sum_{i=1}^n \sum_{f=1}^k v_{if} v_{jf} x_i x_i \right)​$$

$$= \frac{1}{2} \sum_{f=1}^k \left(\sum_{i=1}^n \sum_{j=1}^n v_{if} v_{jf} x_i x_j - \sum_{i=1}^n v_{if} v_{if} x_i x_i \right)​$$

$$= \frac{1}{2}\sum_{f=1}^k\left( \left(\sum_{i=1}^nv_{if} x_i \right) \left( \sum_{j=1}^n v_{jf} x_j \right) - \sum_{i=1}^n \left(v_{if} x_i\right)^2 \right)​$$

$$= \frac{1}{2}\sum_{f=1}^k\left( \left(\sum_{i=1}^nv_{if} x_i \right)^2 - \sum_{i=1}^n \left(v_{if} x_i\right)^2 \right)​$$

$$= \frac{1}{2} \sum_{f=1}^k \left(\left( \sum_{i=1}^n v_{i, f} x_i \right)^2 - \sum_{i=1}^n v_{i, f}^2 x_i^2 \right)​$$

## xLearn的CalcScore实现
```c++
// y = sum( (V_i*V_j)(x_i * x_j) )
// Using SSE to accelerate vector operation.
// row为libsvm格式的样本，采用vector稀疏存储的Node，Node里有feat_id及feat_val
// model为模型
real_t FMScore::CalcScore(const SparseRow* row,
                          Model& model,
                          real_t norm) {
  /*********************************************************
   *  linear term and bias term                            *
   *********************************************************/
  real_t sqrt_norm = sqrt(norm);
  real_t *w = model.GetParameter_w();
  index_t num_feat = model.GetNumFeature();
  real_t t = 0;
  index_t aux_size = model.GetAuxiliarySize();
  for (SparseRow::const_iterator iter = row->begin();
       iter != row->end(); ++iter) {
    index_t feat_id = iter->feat_id;
    // To avoid unseen feature in Prediction
    if (feat_id >= num_feat) continue;
    //计算线性部分x_i*w_i，求和到t
    t += (iter->feat_val * w[feat_id*aux_size] * sqrt_norm);
  }
  // bias
  // 偏置w_0，加到t
  w = model.GetParameter_b();
  t += w[0];
  /*********************************************************
   *  latent factor                                        *
   *********************************************************/
  //隐向量长度调整为4的整数倍aligned_k
  index_t aligned_k = model.get_aligned_k();
  index_t align0 = model.get_aligned_k() * aux_size;
  std::vector<real_t> sv(aligned_k, 0);
  real_t* s = sv.data();
  for (SparseRow::const_iterator iter = row->begin();
       iter != row->end(); ++iter) {
    index_t j1 = iter->feat_id;
    // To avoid unseen feature in Prediction
    if (j1 >= num_feat) continue;
    real_t v1 = iter->feat_val;//x_i
    real_t *w = model.GetParameter_v() + j1 * align0;//v_i
    //SSE指令，x_i存储于128位的寄存器中
    __m128 XMMv = _mm_set1_ps(v1*norm);//x_i
    //循环每次移动4个长度，4个float正好128位，一次循环计算4个浮点数
    for (index_t d = 0; d < aligned_k; d += kAlign) {
      __m128 XMMs = _mm_load_ps(s+d);
      __m128 const XMMw = _mm_load_ps(w+d);//v_i
      //计算v_if * x_i，并按i求和，结果是一个k维向量sv
      XMMs = _mm_add_ps(XMMs, _mm_mul_ps(XMMw, XMMv));
      _mm_store_ps(s+d, XMMs);
    }
  }
  __m128 XMMt = _mm_set1_ps(0.0f);
  for (SparseRow::const_iterator iter = row->begin();
       iter != row->end(); ++iter) {
    index_t j1 = iter->feat_id;
    // To avoid unseen feature in Prediction
    if (j1 >= num_feat) continue;
    real_t v1 = iter->feat_val;//x_i
    real_t *w = model.GetParameter_v() + j1 * align0;//v_i
    //SSE指令，x_i存储于128位的寄存器中
    __m128 XMMv = _mm_set1_ps(v1*norm);
    for (index_t d = 0; d < aligned_k; d += kAlign) {
      __m128 XMMs = _mm_load_ps(s+d);
      __m128 XMMw = _mm_load_ps(w+d);//v_i
      __m128 XMMwv = _mm_mul_ps(XMMw, XMMv);//v_if * x_i
      XMMt = _mm_add_ps(XMMt,
         _mm_mul_ps(XMMwv, _mm_sub_ps(XMMs, XMMwv)));
    }
  }
  XMMt = _mm_hadd_ps(XMMt, XMMt);
  XMMt = _mm_hadd_ps(XMMt, XMMt);
  real_t t_all;
  _mm_store_ss(&t_all, XMMt);
  t_all *= 0.5;
  t_all += t;
  return t_all;
}
```
在xLearn中的实现，并非是论文的简化后的公式，具体如下：

$$\sum_{i=1}^n \sum_{j=i+1}^n \langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j ​$$

$$= \frac{1}{2}\left(\sum_{i=1}^n \sum_{j=1}^n \langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j - \sum_{i=1}^n \langle \mathbf{v}_i, \mathbf{v}_i \rangle x_i x_i \right)​$$

$$= \frac{1}{2}\left(\sum_{i=1}^n \sum_{j=1}^n \sum_{f=1}^k v_{if} v_{jf} x_i x_j - \sum_{i=1}^n \sum_{f=1}^k v_{if} v_{jf} x_i x_i \right)​$$

$$= \frac{1}{2} \sum_{f=1}^k \left(\sum_{i=1}^n \sum_{j=1}^n v_{if} v_{jf} x_i x_j - \sum_{i=1}^n v_{if} v_{if} x_i x_i \right)​$$

$$= \frac{1}{2}\sum_{f=1}^k\left( \sum_{i=1}^nv_{if} x_i\left( \sum_{j=1}^n v_{jf} x_j -  v_{if} x_i\right ) \right)​$$

第一个for循环是计算$\sum_{j=1}^n v_{jf} x_j$，注意下标是j，结果存于sv的vector中，内层嵌套for循环并没有做求和操作，只是遍历的隐向量。

第二个for循环是计算$\sum_{i=1}^n$，注意下标是i，内层嵌套for循环是计算$\sum_{f=1}^k$，被两层循环计算求和的单元是

$$v_{if}x_i\left(\sum_{j=1}^nv_{jf} x_j-v_{if}x_i\right ) $$

用到了前面循环的中间结果，结果存于XMMt，由于内层循环是以4个浮点数同时做计算，所以结果最后需要将这个四个浮点数加起来，调用了两次_mm_hadd_ps实现。



参考链接：

[http://www.algo.uni-konstanz.de/members/rendle/pdf/Rendle2010FM.pdf](http://www.algo.uni-konstanz.de/members/rendle/pdf/Rendle2010FM.pdf)

[https://tech.meituan.com/2016/03/03/deep-understanding-of-ffm-principles-and-practices.html](https://tech.meituan.com/2016/03/03/deep-understanding-of-ffm-principles-and-practices.html)













