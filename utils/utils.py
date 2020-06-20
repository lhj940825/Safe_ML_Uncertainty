import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import yaml
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


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

    figure_dir = os.path.join(figure_dir, title + '_GT-mean_and_std.png')
    plt.savefig(figure_dir)
    plt.show()

def every_10_epochs_plot_scatter2(ground_truth, mean, var, output_dir, title):
    """

    :param ground_truth:
    :param mean:
    :param var:
    :param output_dir:
    :param title:
    :return:
    """

    plt.scatter(ground_truth-mean, np.sqrt(var))
    plt.title(title+ ': GT-mean and std ')
    plt.xlabel('Ground Truth - mean')
    plt.ylabel('std')

    os.makedirs(output_dir, exist_ok=True)
    figure_dir = os.path.join(output_dir, 'figures')
    figure_dir = get_train_or_test_figure_dir(figure_dir, title)
    os.makedirs(figure_dir, exist_ok=True)
    figure_dir = os.path.join(figure_dir, 'GT-mean_and_std_for_each_epoch')
    os.makedirs(figure_dir, exist_ok=True)

    figure_dir = os.path.join(figure_dir, title + '_GT-mean_and_std.png')
    plt.savefig(figure_dir)
    plt.show()




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

def plot_sequence_mean_var(seq_mean, seq_var, output_dir='./tmp_videos', title='sequence'):
    fig = plt.figure(figsize=(8, 6))
    canvas = FigureCanvas(fig)
    w, h = canvas.get_width_height()
    ax = fig.add_subplot(111)
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, title + '.avi')
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (w, h))
    
    max_mean = np.max(seq_mean)
    max_var = np.max(seq_var)
    min_mean = np.min(seq_mean)
    min_var = np.min(seq_var)

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
