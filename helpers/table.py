import pandas as pd
import os.path
import tqdm


try:
    from .images import get_images, get_paeth_images
    from .measures import get_png_size, gini, image_hopkins, shannon_entropy, modified_shannon_entropy, l0_norm,\
            l1_norm, l2_l1_ratio, sparse_log, kurtosis_4, gaussian_entropy, hoyer, sparse_tanh,\
            l0_epsilon, lp_neg, lp_norm, card_image, card_image_mono, dog_l0, greyopening_l1, dog_l1, dog_l2,\
            dog_hs, greyopening_l0, greyopening_l2, greyopening_hs
except:
    from images import get_images, get_paeth_images
    from measures import get_png_size, gini, image_hopkins, shannon_entropy, modified_shannon_entropy, l0_norm,\
            l1_norm, l2_l1_ratio, sparse_log, kurtosis_4, gaussian_entropy, hoyer, sparse_tanh,\
            l0_epsilon, lp_neg, lp_norm, card_image, card_image_mono, dog_l0, greyopening_l1, dog_l1, dog_l2,\
            dog_hs, greyopening_l0, greyopening_l2, greyopening_hs


def measures_table(
        subset='train',
        epsilon_l0eps=0.005,
        p_lp=2,
        a_tanh=0.5,
        b_tanh=2,
        p_lp_neg=0.5
    ):
    """
    Get every measure used in this work, on a subset of CIFAR-100
    Non-parametric measures are precomputed and stored in a CSV file

    Parameters:
        - subset (str, train|test) : subset to precompute measures from
        - epsilon_l0eps : $\epsilon$ to compute $\ell_\epsilon^0$
        - p_lp : $p$ to use to compute $\ell^p$
        - a_tanh : $a$ to compute $tanh_{a, b}$
        - b_tanh : $b$ to compute $tanh_{a, b}$
        - p_lp_neg : $p$ to compute $-\ell^p_-$

    Return :
        - df (pandas.DataFrame) : DataFrame containing every measure (column) for every image (row)
    """
    im_raw   = get_images(subset)
    im_paeth = get_paeth_images(subset)

    df = pd.DataFrame()
    savepath = f"./precomputed/{subset}.csv"
    #if os.path.isfile(savepath):
    #    df = pd.read_csv(savepath)

    # Non Parametric Measures

    if 'PNG Size' not in df.columns:
        print('--- PNG Size ---')
        df['PNG Size'] = [get_png_size(im) for im in tqdm.tqdm(im_raw)]
    
    if '$Hopkins$' not in df.columns:
        print('--- Hopkins Statistic ---')
        df['$Hopkins$'] = [image_hopkins(im) for im in tqdm.tqdm(im_raw)]

    if '$H_S$' not in df.columns:
        print('--- Shannon Entropy ---')
        df['$H_S$'] = [shannon_entropy(im) for im in tqdm.tqdm(im_paeth)]

    if '$H_{S\'}$' not in df.columns:
        print('--- Modified Shannon Entropy ---')
        df['$H_{S\'}$'] = [modified_shannon_entropy(im) for im in tqdm.tqdm(im_paeth)]

    if '$\ell^0$' not in df.columns:
        print('--- L0 Norm ---') 
        df['$\ell^0$'] = [l0_norm(im) for im in tqdm.tqdm(im_paeth)]

    if '$\ell^1$' not in df.columns:
        print('--- L1 Norm ---') 
        df['$\ell^1$'] = [l1_norm(im) for im in tqdm.tqdm(im_paeth)]

    if '$\ell^2$ / $\ell^1$' not in df.columns:
        print('--- L2/L1 Ratio ---') 
        df['$\ell^2$ / $\ell^1$'] = [l2_l1_ratio(im) for im in tqdm.tqdm(im_paeth)]
    
    if '$log$' not in df.columns:
        print('--- Log ---')
        df['$log$'] = [sparse_log(im) for im in tqdm.tqdm(im_paeth)]

    if '$\kappa_4$' not in df.columns:
        print('--- Kurtosis-4 ---')
        df['$\kappa_4$'] = [kurtosis_4(im) for im in tqdm.tqdm(im_paeth)]

    if '$H_G$' not in df.columns:
        print('--- Gaussian Entropy ---')
        df['$H_G$'] = [gaussian_entropy(im) for im in tqdm.tqdm(im_paeth)]

    if '$Hoyer$' not in df.columns:
        print('--- Hoyer ---')
        df['$Hoyer$'] = [hoyer(im) for im in tqdm.tqdm(im_paeth)]

    if '$Gini$' not in df.columns:
        print('--- Gini ---')
        df['$Gini$'] = [gini(im) for im in tqdm.tqdm(im_paeth)]

    if '$Gini_{raw}$' not in df.columns:
        print('--- Raw Gini ---')
        df['$Gini_{raw}$'] = [gini(im) for im in tqdm.tqdm(im_raw)]

    if '$Card$' not in df.columns:
        print('--- Card ---')
        df['$Card$'] = [card_image(im) for im in tqdm.tqdm(im_paeth)]

    if '$Card_{raw}$' not in df.columns:
        print('--- Card ---')
        df['$Card_{raw}$'] = [card_image(im) for im in tqdm.tqdm(im_raw)]  

    if '$Card_{raw}^{mono}$' not in df.columns:
        print('--- Card ---')
        df['$Card_{raw}^{mono}$'] = [card_image_mono(im) for im in tqdm.tqdm(im_raw)]   

    if '$DoG-\ell^0_{raw}$' not in df.columns:
        print('--- DoG raw ---')
        df['$DoG-\ell^0_{raw}$'] = [dog_l0(im) for im in tqdm.tqdm(im_raw)]
    
    if '$DoG-\ell^1_{raw}$' not in df.columns:
        print('--- DoG raw ---')
        df['$DoG-\ell^1_{raw}$'] = [dog_l1(im) for im in tqdm.tqdm(im_raw)]
    
    if '$DoG-\ell^2_{raw}$' not in df.columns:
        print('--- DoG raw ---')
        df['$DoG-\ell^2_{raw}$'] = [dog_l2(im) for im in tqdm.tqdm(im_raw)]
    
    if '$DoG-\ell^H_{raw}$' not in df.columns:
        print('--- DoG raw ---')
        df['$DoG-\ell^H_{raw}$'] = [dog_hs(im) for im in tqdm.tqdm(im_raw)]

    if '$GO-\ell^0_{raw}$' not in df.columns:
        print('--- GO raw ---')
        df['$GO-\ell^0_{raw}$'] = [greyopening_l0(im) for im in tqdm.tqdm(im_raw)]  
    
    if '$GO-\ell^1_{raw}$' not in df.columns:
        print('--- GO raw ---')
        df['$GO-\ell^1_{raw}$'] = [greyopening_l1(im) for im in tqdm.tqdm(im_raw)]  
    
    if '$GO-\ell^2_{raw}$' not in df.columns:
        print('--- GO raw ---')
        df['$GO-\ell^2_{raw}$'] = [greyopening_l2(im) for im in tqdm.tqdm(im_raw)]
    
    if '$GO-\ell^H_{raw}$' not in df.columns:
        print('--- GO raw ---')
        df['$GO-\ell^H_{raw}$'] = [greyopening_hs(im) for im in tqdm.tqdm(im_raw)]  

    if '$DoG-\ell^0$' not in df.columns:
        print('--- DoG raw ---')
        df['$DoG-\ell^0$'] = [dog_l0(im) for im in tqdm.tqdm(im_paeth)]

    if '$DoG-\ell^1$' not in df.columns:
        print('--- DoG raw ---')
        df['$DoG-\ell^1$'] = [dog_l1(im) for im in tqdm.tqdm(im_paeth)]

    if '$DoG-\ell^2$' not in df.columns:
        print('--- DoG raw ---')
        df['$DoG-\ell^2$'] = [dog_l2(im) for im in tqdm.tqdm(im_paeth)]
    
    if '$DoG-\ell^H$' not in df.columns:
        print('--- DoG raw ---')
        df['$DoG-\ell^H$'] = [dog_hs(im) for im in tqdm.tqdm(im_paeth)]

    if '$GO-\ell^0$' not in df.columns:
        print('--- Go raw ---')
        df['$GO-\ell^0$'] = [greyopening_l0(im) for im in tqdm.tqdm(im_paeth)]  

    if '$GO-\ell^1$' not in df.columns:
        print('--- Go raw ---')
        df['$GO-\ell^1$'] = [greyopening_l1(im) for im in tqdm.tqdm(im_paeth)]
    
    if '$GO-\ell^2$' not in df.columns:
        print('--- Go raw ---')
        df['$GO-\ell^2$'] = [greyopening_l2(im) for im in tqdm.tqdm(im_paeth)]  

    if '$GO-\ell^H$' not in df.columns:
        print('--- Go raw ---')
        df['$GO-\ell^H$'] = [greyopening_hs(im) for im in tqdm.tqdm(im_paeth)]  

    #df.to_csv(savepath, index=None)

    # Parametric Measures

    print('--- Hyperbolic Tangent ---')
    df[f'$tanh_{{{a_tanh},{b_tanh}}}$'] = [sparse_tanh(im, a_tanh, b_tanh) for im in tqdm.tqdm(im_paeth)]

    print('--- L0 Epsilon ---')
    df[f'$\ell_{{{epsilon_l0eps}}}^0$'] = [l0_epsilon(im, epsilon_l0eps) for im in tqdm.tqdm(im_paeth)]

    print('--- LP Norm ---', f'$\ell^{{{p_lp}}}$')
    df[f'$\ell^{{{p_lp}}}$'] = [lp_norm(im, p_lp) for im in tqdm.tqdm(im_paeth)]

    print('--- LP with Negative P ---')
    df[f'$-\ell^{{{-p_lp_neg}}}_-$'] = [lp_neg(im, p_lp_neg) for im in tqdm.tqdm(im_paeth)]

    df.to_csv(savepath, index=None)

    return df


if __name__ == '__main__':
    measures_table(subset='train')
    measures_table(subset='test')