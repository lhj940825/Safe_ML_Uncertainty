import numpy as np
import os
import cv2
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import yaml
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

max_epoch = '150'

def split_wine_dataset(wine_data_dir, train_set_dir, test_set_dir):

    if not os.path.exists(wine_data_dir):
        raise FileNotFoundError('{} file is not found. need to download and place the file in the mentioned directory'.format(wine_data_dir))

    else: # when the dataset file 'winequality-white.csv' exists
        if not os.path.exists(train_set_dir): # when the dataset file exists but not has been splitted.
            wine_dataset = pd.read_csv(wine_data_dir, sep=';')
            wine_dataset = (wine_dataset - wine_dataset.min())/(wine_dataset.max()-wine_dataset.min())

            #std_scaler = StandardScaler() # define standard normalizer

            #wine_dataset = pd.DataFrame(std_scaler.fit_transform(wine_dataset), columns=wine_dataset.columns)

            X = wine_dataset.drop(labels= 'quality', axis = 1)


            X = (X-X.min())/(X.max() - X.min()) # min-max normalization

            Y = wine_dataset['quality']

            train_x, test_x, train_y, test_y = train_test_split(X,Y, test_size=0.1, stratify=Y)

            df_train = pd.concat([train_x, train_y], axis=1) # concatanate training set with corresponding labels
            df_test = pd.concat([test_x, test_y], axis=1) # concatanate test set with corresponding labels

            # write splitted dataset to csv
            df_train.to_csv(train_set_dir, index=False)
            df_test.to_csv(test_set_dir, index=False)

def get_args_from_yaml(file):
    with open(file) as f:
        conf = yaml.load(f,Loader=yaml.FullLoader)
    return conf


def draw_loss_trend_figure(title, epoch, train_loss_lost, test_loss_list=None, output_dir=None):
    """
    Draw and Save loss figure for given parameters

    :param title:
    :param train_loss_lost:
    :param test_loss_list:
    :param epoch:
    :return:
    """
    epochs_train = np.arange(0, epoch, 1)

    plt.title(title)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.plot(epochs_train, train_loss_lost, label = 'train loss')
    if test_loss_list is not None:
        epochs_test = np.arange(1, epoch + 1, 1)
        plt.plot(epochs_test, test_loss_list, label= 'validation loss')
    plt.legend(loc='best')

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        figure_dir = os.path.join(output_dir, 'figures')
        os.makedirs(figure_dir, exist_ok=True)

        figure_dir = os.path.join(figure_dir, '{}_loss_fig.png'.format(title))
        plt.savefig(figure_dir)
    plt.show()

def plot_loss(ax, title, epoch, train_loss_lost, test_loss_list=None, output_dir=None):
    epochs_train = np.arange(0, epoch, 1)

    ax.set_title(title)
    ax.set_xlabel('epochs')
    ax.set_ylabel('loss')
    ax.plot(epochs_train, train_loss_lost, label = 'train loss')
    if test_loss_list is not None:
        epochs_test = np.arange(1, epoch + 1, 1)
        plt.plot(epochs_test, test_loss_list, label= 'validation loss')
    plt.legend(loc='best')

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        figure_dir = os.path.join(output_dir, 'figures')
        os.makedirs(figure_dir, exist_ok=True)

        figure_dir = os.path.join(figure_dir, '{}_loss_fig.png'.format(title))
        plt.savefig(figure_dir)
    # plt.show()


def plot_and_save_histograms(NLL_list, RMSE_list, output_dir, title="fig"):
    fig, axs = plt.subplots(1, 2, tight_layout=True)
    num_bins = 60

    # N is the count in each bin, bins is the lower-limit of the bin
    N, bins, patches = axs[0].hist(NLL_list, bins=num_bins, weights=np.ones(len(NLL_list)) / len(NLL_list))
    axs[0].yaxis.set_major_formatter(PercentFormatter(1))

    axs[0].set_title(title + ': NLL Histogram')
    axs[0].set_xlabel('NLL value')
    # We'll color code by height, but you could use any scalar
    fracs = N / N.max()

    # we need to normalize the data to 0..1 for the full range of the colormap
    norm = colors.Normalize(fracs.min(), fracs.max())

    # Now, we'll loop through our objects and set the color of each accordingly
    for thisfrac, thispatch in zip(fracs, patches):
        color = plt.cm.viridis(norm(thisfrac))
        thispatch.set_facecolor(color)


    N, bins, patches = axs[1].hist(RMSE_list, bins=num_bins, weights=np.ones(len(RMSE_list)) / len(RMSE_list))
    axs[1].set_title(title + ': RMSE Histogram')
    axs[1].set_xlabel('RMSE value')
    axs[1].yaxis.set_major_formatter(PercentFormatter(1))

    fracs = N / N.max()
    norm = colors.Normalize(fracs.min(), fracs.max())
    for thisfrac, thispatch in zip(fracs, patches):
        color = plt.cm.viridis(norm(thisfrac))
        thispatch.set_facecolor(color)


    os.makedirs(output_dir, exist_ok=True)
    figure_dir = os.path.join(output_dir, 'figures')
    figure_dir = get_train_or_test_figure_dir(figure_dir, title)
    os.makedirs(figure_dir, exist_ok=True)

    figure_dir = os.path.join(figure_dir, title + '_NLL_RMSE_histogram.png')
    plt.savefig(figure_dir)
    plt.show()

def plot_histograms(data, output_dir=None, title="fig"):
    ax = plt.subplot()
    num_bins = 30

    # N is the count in each bin, bins is the lower-limit of the bin
    N, bins, patches = ax.hist(data, bins=num_bins, weights=np.ones(len(data)) / len(data))
    # ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_yscale('log')
    ax.set_title(title + ': Target Norm Histogram')
    ax.set_xlabel('Target norm')
    # We'll color code by height, but you could use any scalar
    fracs = N / N.max()

    # we need to normalize the data to 0..1 for the full range of the colormap
    norm = colors.Normalize(fracs.min(), fracs.max())

    # Now, we'll loop through our objects and set the color of each accordingly
    for thisfrac, thispatch in zip(fracs, patches):
        color = plt.cm.viridis(norm(thisfrac))
        thispatch.set_facecolor(color)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        figure_dir = os.path.join(output_dir, 'figures')
        figure_dir = get_train_or_test_figure_dir(figure_dir, title)
        os.makedirs(figure_dir, exist_ok=True)

        figure_dir = os.path.join(figure_dir, title + '_target_norm_histogram.png')
        plt.savefig(figure_dir)
    plt.show()

def plot_scatter(NLL_list, RMSE_list, output_dir, title="fig"):
    plt.scatter(NLL_list, RMSE_list)
    plt.title(title + ': NLL and RMSE Scatter')
    plt.xlabel('NLL value')
    plt.ylabel('RMSE value')


    os.makedirs(output_dir, exist_ok=True)
    figure_dir = os.path.join(output_dir, 'figures')
    figure_dir = get_train_or_test_figure_dir(figure_dir, title)
    os.makedirs(figure_dir, exist_ok=True)

    figure_dir = os.path.join(figure_dir, title + '_NLL_RMSE_scatter.png')
    plt.savefig(figure_dir)
    plt.show()

def plot_scatter2(ground_truth, mean, var, output_dir, title):
    """
    scatter figure with 'ground_truth - mean' as x-axis and 'std' as y-axis

    :param ground_truth: target label
    :param mean: computed mean from network samples
    :param var: computed variance
    :param output_dir:
    :param title:
    :return:
    """
    #print('gt',len(ground_truth), np.shape(ground_truth))
    #print('mean',len(mean), np.shape(mean))
    #print('var',len(var), np.shape(var))
    plt.scatter(ground_truth-mean, np.sqrt(var))
    plt.title(title+ ': GT-mean and std ')
    plt.xlabel('Ground Truth - mean')
    plt.ylabel('std')


    os.makedirs(output_dir, exist_ok=True)
    figure_dir = os.path.join(output_dir, 'figures')
    figure_dir = get_train_or_test_figure_dir(figure_dir, title)
    os.makedirs(figure_dir, exist_ok=True)

    figure_dir = os.path.join(figure_dir, title + ', GT-mean_and_std.png')
    plt.savefig(figure_dir)
    plt.show()

def residual_error_and_std_plot_with_y_equal_abs_x_graph(residual_error, std, output_dir, title, y_axis_contraint, y_max=None, denormalized = None):
    """
    plot the 'gt-mean and std' figure (named as scatter2 in other functions) with y=abs(x) graph

    :param residual_error: Ground truth - mean
    :param std: standard deviation
    :param output_dir:
    :param title:
    :param y_axis_contraint: if yes, range of y is [0,3]
    :return:
    """

    plt.scatter(residual_error, std)
    plt.title(title+ ': GT-mean and std ')
    plt.xlabel('Ground Truth - mean')
    plt.ylabel('std')
    #plt.ylim(0,5)

    x = np.linspace(0, np.max(residual_error), 100)
    plt.plot(x, x,'r-', lw=5, alpha=0.6, label='y=x')
    plt.plot(-x, x,'r-', lw=5, alpha=0.6)
    plt.legend()

    os.makedirs(output_dir, exist_ok=True)
    figure_dir = os.path.join(output_dir, 'figures')
    figure_dir = get_train_or_test_figure_dir(figure_dir, title)
    os.makedirs(figure_dir, exist_ok=True)

    if not y_axis_contraint: # When y axis range is not contrained
        image_folder_dir = os.path.join(figure_dir, 'GT-mean_and_std_for_each_epoch')
        video_title = title[:title.find('=')] +'_GT-mean_and_std'
    else: # when y axis is contrained

        plt.ylim(ymin=0, ymax=y_max)
        image_folder_dir = os.path.join(figure_dir, 'GT-mean_and_std_for_each_epoch_with_y_lim')
        video_title = title[:title.find('=')] +'_GT-mean_and_std_with_y_lim'

    os.makedirs(image_folder_dir, exist_ok=True)
    figure_dir = os.path.join(image_folder_dir, title + ', GT-mean_and_std.png')

    #os.makedirs(figure_dir, exist_ok=True)
    plt.savefig(figure_dir)
    plt.show()

    if max_epoch in title: # if there are generated figures from epoch 0 to epoch max then generate the video out of those figures
        save_histograms_and_scatter2_variants_videos(image_folder_dir, video_title)


def residual_error_and_std_plot_with_NLL_heatmap(residual_error, std, output_dir, title, y_max, residual_error_max, denormalized = None):
    """
    plot the 'gt-mean and std' figure (named as scatter2 in other functions) with y=abs(x) graph and NLL heatmap

    :param residual_error: Ground truth - mean
    :param std:
    :param output_dir:
    :param title:
    :param y_max:
    :param residual_error_max: maximum value in residual error
    :return:
    """

    x_coordinates = np.linspace(-residual_error_max, residual_error_max, 100)
    y_coordinates = np.linspace(0.01, y_max, 100)

    NLL_values = []

    np.seterr(invalid='ignore') #ignoring the warning
    for y_coordinate in y_coordinates:
        for x_coordinate in x_coordinates:

            NLL_values.append(np.log(compute_NLL(x_coordinate, y_coordinate)))
    np.seterr(invalid='warn') #set numpy not to ignore warning


    NLL_values = np.reshape(np.asarray(NLL_values), (len(y_coordinates), -1))
    NLL_values = DataFrame(NLL_values, columns=x_coordinates, index=y_coordinates)
    #NLL_values=(NLL_values-NLL_values.mean())/NLL_values.std()

    pos = plt.pcolor(x_coordinates, y_coordinates, NLL_values)
    cbar = plt.colorbar(pos)
    cbar.set_label("log(NLL)")

    #plot network outputs (GT-mean, std)
    plt.scatter(residual_error, std)
    plt.title(title+ ': GT-mean and std ')
    plt.xlabel('Ground Truth - mean')
    plt.ylabel('std')

    #plot y=x
    x = np.linspace(0, np.max(residual_error), 100)
    plt.plot(x, x,'r-', lw=5, alpha=0.6, label='y=x')
    plt.plot(-x, x,'r-', lw=5, alpha=0.6)
    plt.legend()
    plt.ylim(ymin=0, ymax=y_max)

    figure_dir = os.path.join(output_dir, 'figures')
    figure_dir = get_train_or_test_figure_dir(figure_dir, title)
    os.makedirs(figure_dir, exist_ok=True)
    image_folder_dir = os.path.join(figure_dir, 'GT-mean_and_std_for_each_epoch_with_y_lim_and_heatmap')
    os.makedirs(image_folder_dir, exist_ok=True)

    figure_dir = os.path.join(image_folder_dir, title + ', GT-mean_and_std.png')
    plt.savefig(figure_dir)
    plt.show()


    if max_epoch in title: # if there are generated figures from epoch 0 to epoch max then generate the video out of those figures
        save_histograms_and_scatter2_variants_videos(image_folder_dir, title[:title.find('=')] +'_GT-mean_and_std_with_y_lim_and_heatmap' )


def plot_NLL_histogram(NLL_list, output_dir, title):
    """
    plot the histogram of given list of NLL values

    :param NLL_list:
    :param output_dir:
    :param title:
    :return:
    """
    fig, axs = plt.subplots(1, 1, tight_layout=True)
    num_bins = 60

    # N is the count in each bin, bins is the lower-limit of the bin
    N, bins, patches = axs.hist(NLL_list, bins=num_bins, weights=np.ones(len(NLL_list)) / len(NLL_list))
    axs.yaxis.set_major_formatter(PercentFormatter(1))

    axs.set_title(title + ': NLL Histogram')
    axs.set_xlabel('NLL value without v-noise')
    # We'll color code by height, but you could use any scalar
    fracs = N / N.max()

    # we need to normalize the data to 0..1 for the full range of the colormap
    norm = colors.Normalize(fracs.min(), fracs.max())

    # Now, we'll loop through our objects and set the color of each accordingly
    for thisfrac, thispatch in zip(fracs, patches):
        color = plt.cm.viridis(norm(thisfrac))
        thispatch.set_facecolor(color)

    figure_dir = os.path.join(output_dir, 'figures')
    figure_dir = get_train_or_test_figure_dir(figure_dir, title)
    os.makedirs(figure_dir, exist_ok=True)
    image_folder_dir = os.path.join(figure_dir, 'NLL_histograms_per_epoch')
    os.makedirs(image_folder_dir, exist_ok=True)

    figure_dir = os.path.join(image_folder_dir, title + ', NLL_histogram_without_v_noise.png')
    plt.savefig(figure_dir)
    plt.show()

    if max_epoch in title: # if there are generated figures from epoch 0 to epoch max then generate the video out of those figures
        save_histograms_and_scatter2_variants_videos(image_folder_dir, title[:title.find('=')] +'_NLL_histogram_without_v_noise' )


def plot_scatter2_and_NLL_histogram_variants(ground_truth, mean, var, output_dir, title):
    """
    plot many variants of the 'ground truth - mean and std' figures and NLL histogram, e.g with y=x graph, with NLL heatmap, with y-axis contraint

    :param ground_truth:
    :param mean:
    :param var:
    :param output_dir:
    :param title:
    :return:
    """

    std = np.sqrt(var)
    gt_sub_mean = ground_truth-mean
    y_max = 3 # maximum value in y_axis range
    gt_sub_mean_max = abs(np.min(gt_sub_mean)) if abs(np.min(gt_sub_mean)) > abs(np.max(gt_sub_mean)) else abs(np.max(gt_sub_mean))

    for flag in [True, False]: # save figures with normalized
        residual_error_and_std_plot_with_y_equal_abs_x_graph(gt_sub_mean,std,output_dir,title, y_axis_contraint=False, y_max=None, denormalized=flag)
        residual_error_and_std_plot_with_y_equal_abs_x_graph(gt_sub_mean,std,output_dir,title, y_axis_contraint=True, y_max=y_max, denormalized=flag)
        residual_error_and_std_plot_with_NLL_heatmap(gt_sub_mean,std,output_dir,title, y_max, gt_sub_mean_max, denormalized = flag)


    NLL = compute_NLL(gt_sub_mean, std)
    plot_NLL_histogram(NLL, output_dir, title)
    #


def compute_NLL(gt_sub_mean, std):
    if isinstance(std, np.float64):
        if std == 0:
            std = 1e-6
    else: # when std is an array from numpy
        std[std==0] = 1e-6

    a = np.log(std**2)*0.5
    b = np.divide(np.square(gt_sub_mean), (2*(std**2)))
    NLL = a+b
    return NLL


def plot_Mahalanobis_distance(sample_M_distance_list, gt_M_distance_list, output_dir, title="fig"):
    num_bins = 100
    bins=np.histogram(np.hstack((sample_M_distance_list,gt_M_distance_list)), bins=num_bins)[1] # to get the equal bin width

    plt.hist(gt_M_distance_list, bins,   alpha=0.5, label='Ground Truth',  weights=np.ones(len(gt_M_distance_list)) / len(gt_M_distance_list))
    plt.hist(sample_M_distance_list, bins,  alpha=0.5, label='Sample', weights=np.ones(len(sample_M_distance_list)) / len(sample_M_distance_list))

    # Logarithmic y-axis bins
    plt.yscale('log', nonposy='clip')

    plt.legend(loc='upper right')
    plt.xlabel('Squared Mahalanobis_distance')
    plt.ylabel('rel frequency(log scale)')
    plt.title(title + ': Assessment of uncertainty realism')


    os.makedirs(output_dir, exist_ok=True)
    figure_dir = os.path.join(output_dir, 'figures')
    figure_dir = get_train_or_test_figure_dir(figure_dir, title)
    os.makedirs(figure_dir, exist_ok=True)
    figure_dir = os.path.join(figure_dir, title + '_Assesment of Uncertainty Realism.png')
    plt.savefig(figure_dir)

    plt.show()

def plot_Mahalanobis_distance_with_Chi2_PDF(sample_M_distance_list, output_dir, title="fig"):
    from scipy.stats import chi2

    x = np.linspace(chi2.ppf(0.01, df=1),chi2.ppf(0.999999, df=1), 1000)
    plt.plot(x, chi2.pdf(x, df=1),'b-', lw=5, alpha=0.6, label='chi2 pdf')

    num_bins = 200

    plt.hist(sample_M_distance_list, bins=num_bins, color='sandybrown',  alpha=0.5, label='Sample', density=True)
    #plt.hist(chi, bins=bins,  alpha=0.5, label='Chi PDF', density=True)

    plt.legend(loc='upper right')
    plt.xlabel('Squared Mahalanobis_distance')
    plt.ylabel('PDF(log scale)')
    plt.xlim(0, max(sample_M_distance_list))

    plt.title(title + ': Assessment of uncertainty realism with Chi PDF')

    # Logarithmic y-axis bins
    plt.yscale('log', nonposy='clip')


    os.makedirs(output_dir, exist_ok=True)
    figure_dir = os.path.join(output_dir, 'figures')
    figure_dir = get_train_or_test_figure_dir(figure_dir, title)
    os.makedirs(figure_dir, exist_ok=True)
    figure_dir = os.path.join(figure_dir, title + '_Assesment of Uncertainty Realism with Chi PDF.png')
    plt.savefig(figure_dir)

    plt.show()

def plot_sequence_mean_var(seq_mean, seq_var, xy_lim=None, output_dir='./tmp_videos', title='sequence'):
    fig = plt.figure(figsize=(8, 6))
    canvas = FigureCanvas(fig)
    w, h = canvas.get_width_height()
    ax = fig.add_subplot(111)
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, title + '.avi')
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (w, h))

    inf = 1e10

    if xy_lim is None:
        max_mean = min(inf, np.max(seq_mean))
        max_var = min(inf, np.max(seq_var))
        min_mean = max(-inf, np.min(seq_mean))
        min_var = max(-inf, np.min(seq_var))
    else:
        max_mean = xy_lim[0]
        max_var = xy_lim[1]
        min_mean = -xy_lim[0]
        min_var = 0.0

    for epoch, (mean, var) in enumerate(zip(seq_mean, seq_var)):
        ax.cla()

        ax.set_xlim(min_mean, max_mean)
        ax.set_ylim(min_var, max_var)

        ax.set_title('{} Epoch: {}'.format(title, epoch))
        ax.set_xlabel('GT - mean')
        # ax.axis("off")

        ax.scatter(mean, var, s=2, c='blue')

        canvas.draw()
        img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(h, w, -1)
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        writer.write(bgr)

    plt.close(fig)
    writer.release()

def plot_NLL_cap_cnt(dataset_list, NLL_cap_cnt, cap, output_dir):
    """
    plot bar graph showing the number of NLL values beyond the cap for all datasets

    :return:
    """

    y_pos = np.arange(len(dataset_list))
    plt.bar(y_pos, NLL_cap_cnt, align='center', alpha=0.5)
    plt.xticks(y_pos, dataset_list)
    plt.ylabel('#NLL values beyond the cap')
    plt.title('Counter of NLL values higher than cap: ' + str(cap))
    figure_dir = os.path.join(output_dir, 'NLL_counter_cap ' + str(cap)+'.png')
    plt.savefig(figure_dir)

    plt.show()

    #print(dataset_list, NLL_cap_cnt, cap)
    #pass


def get_train_or_test_figure_dir(output_dir, title):
    if 'train' in title: # when the figures are from training dataset
        figure_dir = os.path.join(output_dir, 'train')

    elif 'test': # from test dataset
        figure_dir = os.path.join(output_dir, 'test')

    else:
        raise NameError('please make sure the name of your variable title starts with train or test')

    os.makedirs(figure_dir, exist_ok=True)
    return figure_dir

def store_train_mean_and_std(dataset_name,mean: np.float64,std):
    """
    Store the computed mean and standard deviation on whole training set in yml file.
    These mean and std will be used to compute the NLL as v-noise.


    :param dataset_name: the name of training dataset that is used to compute mean and std
    :param mean: computed mean of training dataset
    :param std: computed std of training dataset
    :return:
    """
    import yaml

    yml_dir = os.path.join(os.getcwd(), 'configs/mean_std.yml')
    stream = open(yml_dir, 'r')
    data = yaml.load(stream,Loader=yaml.BaseLoader)

    data[dataset_name]['mean'] = mean.item()
    data[dataset_name]['std'] = std.item()

    with open(yml_dir, 'w') as f:
        yaml.dump(data, f)


def save_histograms_and_scatter2_variants_videos(image_folder, title):
    video_folder = os.path.dirname(image_folder)
    video_folder = os.path.join(video_folder, 'videos')
    os.makedirs(video_folder, exist_ok=True)


    video_name = title+'.avi'
    video_name = os.path.join(video_folder, video_name)
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort(key=lambda f: int(f[f.find('=')+1: f.find(',')]))  # sort images to make those in right order: from epoch 0 to epoch 150

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 1, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()

"""
def plot_Mahalanobis_distance(sample_M_distance_list, gt_M_distance_list, output_dir, title="fig"):
    
    num_bins = 100

    bins=np.histogram(np.hstack((sample_M_distance_list,gt_M_distance_list)), bins=num_bins)[1] # to get the equal bin width

    plt.hist(gt_M_distance_list, bins,   alpha=0.5, label='Ground Truth', weights=np.ones(len(gt_M_distance_list)) / len(gt_M_distance_list))
    plt.hist(sample_M_distance_list, bins,  alpha=0.5, label='Sample', weights=np.ones(len(sample_M_distance_list)) / len(sample_M_distance_list))


    plt.legend(loc='upper right')
    plt.xlabel('Squared Mahalanobis_distance')
    plt.ylabel('frequency')
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.title(title + ': Assessment of uncertainty realism')

    os.makedirs(output_dir, exist_ok=True)
    figure_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figure_dir, exist_ok=True)
    figure_dir = os.path.join(figure_dir, title + '_Assesment of Uncertainty Realism.png')
    plt.savefig(figure_dir)

    plt.show()
"""
