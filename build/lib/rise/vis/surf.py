import scipy.io as sio
import numpy as np
import os
import nibabel as nib 
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image, ImageOps
import glob
import nibabel as nib
import nibabel.gifti
import nilearn
from nilearn import datasets, plotting
import pandas as pd
from matplotlib.colors import ListedColormap
import matplotlib.colors as colors
mpl.rcParams['svg.fonttype'] = 'none'
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO


def _load_image_silently(filename):
    """Load an image without emitting recoverable header-fix messages."""
    logger = nib.imageglobals.logger
    previous_level = logger.level
    logger.setLevel(40)
    try:
        return nib.load(filename)
    finally:
        logger.setLevel(previous_level)


def get_colorbar_and_data_ranges(
    stat_map_data,
    vmin = None,
    vmax=None,
    symmetric_cbar=True,
    force_min_stat_map_value=None
):

    # avoid dealing with masked_array:
    if hasattr(stat_map_data, "_mask"):
        stat_map_data = np.asarray(
            stat_map_data[np.logical_not(stat_map_data._mask)]
        )

    if force_min_stat_map_value is None:
        stat_map_min = np.nanmin(stat_map_data)
    else:
        stat_map_min = force_min_stat_map_value
    stat_map_max = np.nanmax(stat_map_data)

    if symmetric_cbar == "auto":
        if (vmin is None) or (vmax is None):
            symmetric_cbar = stat_map_min < 0 and stat_map_max > 0
        else:
            symmetric_cbar = np.isclose(vmin, -vmax)

    # check compatibility between vmin, vmax and symmetric_cbar
    if symmetric_cbar:
        if vmin is None and vmax is None:
            vmax = max(-stat_map_min, stat_map_max)
            vmin = -vmax
        elif vmin is None:
            vmin = -vmax
        elif vmax is None:
            vmax = -vmin
        elif not np.isclose(vmin, -vmax):
            raise ValueError(
                "vmin must be equal to -vmax unless symmetric_cbar is False."
            )
        cbar_vmin = vmin
        cbar_vmax = vmax

    # set colorbar limits
    else:
        negative_range = stat_map_max <= 0
        positive_range = stat_map_min >= 0
        if positive_range:
            if vmin is None:
                cbar_vmin = 0
            else:
                cbar_vmin = vmin
            cbar_vmax = vmax
        elif negative_range:
            if vmax is None:
                cbar_vmax = 0
            else:
                cbar_vmax = vmax
            cbar_vmin = vmin
        else:
            # limit colorbar to plotted values
            cbar_vmin = vmin
            cbar_vmax = vmax

    # set vmin/vmax based on data if they are not already set
    if vmin is None:
        vmin = stat_map_min
    if vmax is None:
        vmax = stat_map_max

    return cbar_vmin, cbar_vmax, vmin, vmax


def plot_surf_stat_map(coords, faces, stat_map=None,
        elev=0, azim=0,
        cmap='jet',
        threshold=None, bg_map=None,
        mask=None,
        bg_on_stat=False,
        alpha='auto',darkness=0.5,vmin = None,
        vmax=None, symmetric_cbar="auto", returnAx=False,
        figsize=(14,11), label=None, lenient=None,
        **kwargs):

    ''' Visualize results on cortical surface using matplotlib'''
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.tri as tri
    from mpl_toolkits.mplot3d import Axes3D

    # load mesh and derive axes limits
    faces = np.array(faces, dtype=int)
    limits = [coords.min(), coords.max()]

    # set alpha if in auto mode
    if alpha == 'auto':
        if bg_map is None:
            alpha = .5
        else:
            alpha = 1

    # if cmap is given as string, translate to matplotlib cmap
    if type(cmap) == str:
        cmap = plt.cm.get_cmap(cmap)

    # initiate figure and 3d axes
    if figsize is not None:
        fig = plt.figure(figsize=figsize)
    else:
        fig = plt.figure()

    fig.patch.set_facecolor('white')
    ax1 = fig.add_subplot(111, projection='3d', xlim=limits, ylim=limits)
    # ax1._axis3don = False
    ax1.grid(False)
    ax1.set_axis_off()
    ax1.w_zaxis.line.set_lw(0.)
    ax1.set_zticks([])
    ax1.view_init(elev=elev, azim=azim)
    
    # plot mesh without data
    p3dcollec = ax1.plot_trisurf(coords[:, 0], coords[:, 1], coords[:, 2],
                                triangles=faces, linewidth=0.,
                                antialiased=False,
                                color='white')

    
    if mask is not None:
        cmask = np.zeros(len(coords))
        cmask[mask] = 1
        cutoff = 2
        if lenient:
            cutoff = 0
        fmask = np.where(cmask[faces].sum(axis=1) > cutoff)[0]
        
    # If depth_map and/or stat_map are provided, map these onto the surface
    # set_facecolors function of Poly3DCollection is used as passing the
    # facecolors argument to plot_trisurf does not seem to work
    if bg_map is not None or stat_map is not None:

        face_colors = np.ones((faces.shape[0], 4))
        face_colors[:, :3] = .5*face_colors[:, :3]

        if bg_map is not None:
            bg_data = bg_map
            if bg_data.shape[0] != coords.shape[0]:
                raise ValueError('The bg_map does not have the same number '
                                 'of vertices as the mesh.')
            bg_faces = np.mean(bg_data[faces], axis=1)
            bg_faces = bg_faces - bg_faces.min()
            bg_faces = bg_faces / bg_faces.max()
            face_colors = plt.cm.gray_r(bg_faces*darkness)


        # modify alpha values of background
        face_colors[:, 3] = alpha*face_colors[:, 3]

        if stat_map is not None:
            stat_map_data = stat_map
            stat_map_faces = np.mean(stat_map_data[faces], axis=1)
            if label:
                stat_map_faces = np.median(stat_map_data[faces], axis=1)

            # Call _get_plot_stat_map_params to derive symmetric vmin and vmax
            # And colorbar limits depending on symmetric_cbar settings
            cbar_vmin, cbar_vmax, vmin, vmax = \
                get_colorbar_and_data_ranges(stat_map_faces, vmin,vmax,symmetric_cbar)
  
            if threshold is not None:
                kept_indices = np.where(abs(stat_map_faces) >= threshold)[0]
                stat_map_faces = stat_map_faces - vmin
                stat_map_faces = stat_map_faces / (vmax-vmin)
                if bg_on_stat:
                    face_colors[kept_indices] = cmap(stat_map_faces[kept_indices]) * face_colors[kept_indices]
                else:
                    face_colors[kept_indices] = cmap(stat_map_faces[kept_indices])
            else:
                stat_map_faces = stat_map_faces - vmin
                stat_map_faces = stat_map_faces / (vmax-vmin)
                if bg_on_stat:
                    if mask is not None:
                        face_colors[fmask,:] = cmap(stat_map_faces)[fmask,:] * face_colors[fmask,:]
                    else:
                        face_colors = cmap(stat_map_faces) * face_colors
                else:
                    face_colors = cmap(stat_map_faces)

        p3dcollec.set_facecolors(face_colors)
    
    if returnAx == True:
        return fig, ax1
    else:
        return fig,vmin,vmax


def showSurf(input_data, surf, sulc, cort,dpi, showall=None, output_file=None, cmap='jet', symmetric_cbar=True,vmin = None,vmax=None, darkness=0.5,threshold=None, boundary=False, boundary_color='#626262'):    

    """
    Visualize surface statistical maps using the provided input data and surface mesh information.

    Parameters:
    - input_data (numpy.ndarray): The statistical data to be mapped onto the surface.
    - surf (list): A list containing surface mesh information. The first element is the array of vertex coordinates, and the second element is the array of faces (triangles).
    - sulc (numpy.ndarray): Sulcal depth values for the surface vertices.
    - cort (numpy.ndarray): A mask indicating the cortical vertices.
    - dpi (int): The resolution of the output image.
    - showall (bool, optional): If True, show views from multiple angles.
    - output_file (str, optional): The base name of the output files. If provided, images will be saved.
    - cmap (str, optional): The colormap to be used for the statistical map. Default is 'jet'.
    - symmetric_cbar (bool, optional): Whether to make the colorbar symmetric. Default is True.
    - vmin (float, optional): The minimum value for the colormap. If None, it will be set automatically.
    - vmax (float, optional): The maximum value for the colormap. If None, it will be set automatically.
    - darkness (float, optional): The darkness of the background surface. Default is 0.5.
    - threshold (float, optional): The threshold to apply to the statistical map.
    - boundary (numpy.ndarray, optional): The boundary information for the surface.
    - boundary_color (str, optional): The color to be used for the boundary. Default is '#626262'.
    """

    single_color_cmap = LinearSegmentedColormap.from_list("single_color", [boundary_color, boundary_color], N=256)
    
    f,vmin_,vmax_ = plot_surf_stat_map(surf[0], surf[1], bg_map=sulc, mask=cort, stat_map=input_data, bg_on_stat=True, azim=0, cmap=cmap,vmin = vmin,vmax=vmax,
                          symmetric_cbar=symmetric_cbar, threshold=threshold,darkness = darkness)
    if boundary:
        plotting.plot_surf_contours([surf[0], surf[1]], input_data, figure=f, cmap=single_color_cmap)
    #plt.show()
    plt.close()

    if output_file:
        count = 0
        f.savefig((output_file + '.%s.png') % str(count), dpi=dpi)
        plt.tight_layout()
        count += 1
    f,_,_ = plot_surf_stat_map(surf[0], surf[1], bg_map=sulc, mask=cort, stat_map=input_data, bg_on_stat=True, azim=180, cmap=cmap,vmin = vmin,vmax=vmax,
                          symmetric_cbar=symmetric_cbar, threshold=threshold,darkness = darkness)
    if boundary:
        plotting.plot_surf_contours([surf[0], surf[1]], input_data, figure=f, cmap=single_color_cmap)
    #plt.show()
    plt.close()

    if output_file:
        f.savefig((output_file + '.%s.png') % str(count), dpi=dpi)
        count += 1
    if showall:
        f,vmin_,vmax_ = plot_surf_stat_map(surf[0], surf[1], bg_map=sulc, mask=cort, stat_map=input_data, bg_on_stat=True, azim=90, cmap=cmap,vmin = vmin,vmax=vmax,
                              symmetric_cbar=symmetric_cbar, threshold=threshold,darkness = darkness)
        plt.show()
        if output_file:
            f.savefig((output_file + '.%s.png') % str(count), dpi=dpi)
            count += 1
        f,_,_ = plot_surf_stat_map(surf[0], surf[1], bg_map=sulc, mask=cort, stat_map=input_data, bg_on_stat=True, azim=270, cmap=cmap,vmin = vmin,vmax=vmax,
                              symmetric_cbar=symmetric_cbar, threshold=threshold,darkness = darkness)
        plt.show()
        if output_file:
            f.savefig((output_file + '.%s.png') % str(count), dpi=dpi)
            count += 1
        f,_,_ = plot_surf_stat_map(surf[0], surf[1], bg_map=sulc, mask=cort, stat_map=input_data, bg_on_stat=True, elev=90, cmap=cmap,vmin = vmin,vmax=vmax,
                              symmetric_cbar=symmetric_cbar, threshold=threshold,darkness = darkness)
        plt.show()
        if output_file:
            f.savefig((output_file + '.%s.png') % str(count), dpi=dpi)
            count += 1
        f,_,_ = plot_surf_stat_map(surf[0], surf[1], bg_map=sulc, mask=cort, stat_map=input_data, bg_on_stat=True, elev=270, cmap=cmap,vmin = vmin,vmax=vmax,
                              symmetric_cbar=symmetric_cbar, threshold=threshold,darkness = darkness)
        plt.show()
        if output_file:
            f.savefig((output_file + '.%s.png') % str(count), dpi=dpi)
            count += 1

    return vmin_,vmax_



def imageCrop(filename):

    from PIL import Image

    i1 = Image.open(filename)
    i2 = np.array(i1)
    i2[i2.sum(axis=2) == 255*4,:] = 0
    i3 = i2.sum(axis=2)
    x = np.where((i3.sum(axis=1) != 0) * 1)[0]
    y = np.where((i3.sum(axis=0) != 0) * 1)[0]

    result = Image.fromarray(i2[x.squeeze()][:,y.squeeze()])
    result.save(filename)
    
    
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resources')
path_global = path


surfmL = _load_image_silently(os.path.join(path, 'S1200.L.midthickness_MSMAll.32k_fs_LR.surf.gii')).darrays
surfiL = _load_image_silently(os.path.join(path, 'S1200.L.very_inflated_MSMAll.32k_fs_LR.surf.gii')).darrays
surfL = []
surfL.append(np.array(surfmL[0].data*0.3 + surfiL[0].data*0.7))
surfL.append(np.array(surfmL[1].data))

surfmR = _load_image_silently(os.path.join(path, 'S1200.R.midthickness_MSMAll.32k_fs_LR.surf.gii')).darrays
surfiR = _load_image_silently(os.path.join(path, 'S1200.R.very_inflated_MSMAll.32k_fs_LR.surf.gii')).darrays
surfR = []
surfR.append(np.array(surfmR[0].data*0.3 + surfiR[0].data*0.7))
surfR.append(np.array(surfmR[1].data))
                                      
res = _load_image_silently(os.path.join(path_global, 'L.atlasroi.32k_fs_LR.shape.gii'))
res = res.darrays[0].data
cortL = np.squeeze(np.array(np.where(res != 0)[0], dtype=np.int32))
                                      
res = _load_image_silently(os.path.join(path_global, 'R.atlasroi.32k_fs_LR.shape.gii'))
res = res.darrays[0].data
cortR = np.squeeze(np.array(np.where(res != 0)[0], dtype=np.int32))
cortLen = len(cortL) + len(cortR)
del res

sulcL = np.zeros(len(surfL[0]))
sulcR = np.zeros(len(surfR[0]))
sulc_image = _load_image_silently(os.path.join(path, 'S1200.sulc_MSMAll.32k_fs_LR.dscalar.nii'))
sulc_data = np.array(sulc_image.dataobj)
sulcL[cortL] = -1 * sulc_data[0, :len(cortL)]
sulcR[cortR] = -1 * sulc_data[0, len(cortL)::]
medial_sulc = np.array(
    _load_image_silently(
        os.path.join(path, 'Q1-Q6_R440.sulc.32k_fs_LR.dscalar.nii')
    ).dataobj
).squeeze()
sulcL[np.setdiff1d(range(32492),cortL)] = -1 * medial_sulc[np.setdiff1d(range(32492),cortL)]
sulcR[np.setdiff1d(range(32492),cortR)] = -1 * medial_sulc[32492+np.setdiff1d(range(32492),cortR)]


def PNGWhiteTrim(input_png):
    image=Image.open(input_png)
    image.load()
    imageSize = image.size

    # remove alpha channel
    invert_im = image.convert("RGB") 

    # invert image (so that white is 0)
    invert_im = ImageOps.invert(invert_im)
    imageBox = invert_im.getbbox()

    cropped=image.crop(imageBox)
    return cropped


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

cmap = plt.get_cmap('nipy_spectral')
new_cmap = truncate_colormap(cmap, 0.2, 0.95)

#colors1 = plt.cm.YlGnBu(np.linspace(0, 1, 128))
first = int((128*2)-np.round(255*(1.-0.90)))
second = (256-first)
#colors2 = new_cmap(np.linspace(0, 1, first))
colors2 = plt.cm.viridis(np.linspace(0.1, .98, first))
colors3 = plt.cm.YlOrBr(np.linspace(0.25, 1, second))
colors4 = plt.cm.PuBu(np.linspace(0., 0.5, second))
#colors4 = plt.cm.pink(np.linspace(0.9, 1., second))
# combine them and build a new colormap
cols = np.vstack((colors2,colors3))
mymap = colors.LinearSegmentedColormap.from_list('my_colormap', cols)


def visualize_surface_32k_fs_LR(save_path, name, dpi, mymap=mymap, Data=None, dataL=None, dataR=None, vmax=None, threshold=None, darkness =None, boundary=False, Sym = False,boundary_color='#626262'):
    """
    Function to visualize and save surface brain maps with optional boundary overlays and colorbars.

    Parameters:
    - save_path (str): The path where the resulting image will be saved.
    - name (str): The title of the resulting image.
    - dpi (int): The resolution of the saved image in dots per inch.
    - mymap (colormap): The colormap used for the surface plots. Default is 'mymap'.
    - Data (array-like, optional): The combined data for both left and right hemispheres.
    - dataL (array-like, optional): Data specific to the left hemisphere.
    - dataR (array-like, optional): Data specific to the right hemisphere.
    - vmax (float, optional): The maximum value for the color scale. If None, it's calculated based on the 95th percentile.
    - threshold (float, optional): The value below which data is not displayed.
    - darkness (float, optional): The darkness level for the background surface.
    - boundary (bool, optional): If True, a boundary overlay is added to the surface plots.
    - Sym (bool, optional): If True, the color scale is symmetric around zero.
    - boundary_color (str, optional): The color of the boundary overlay. Default is '#626262'.

    Created by Ang Li, modified by Tongyu Zhang. Date 2024.7.24
    """
        
    if Data is not None:
       
        dataL = np.zeros(len(surfL[0]))
        dataL[cortL] = Data[0:len(cortL)]

        dataR = np.zeros(len(surfR[0]))
        dataR[cortR] = Data[len(cortL):cortLen]

    if vmax==None:
        vmax = np.mean(np.percentile(dataL,95)+ np.percentile(dataR,95))

    vmin_left,vmax_left = showSurf(dataL, surfL, sulcL, cortL, dpi,showall=None, output_file=os.path.join(save_path, 'fig.hcp.embed.L'), symmetric_cbar = Sym,vmax =vmax,
             cmap=mymap, threshold=threshold, darkness = darkness,boundary=boundary, boundary_color=boundary_color)
    vmin_right,vmax_right = showSurf(dataR, surfR, sulcR, cortR, dpi,showall=None, output_file=os.path.join(save_path, 'fig.hcp.embed.R'), symmetric_cbar = Sym,vmin = vmin_left,vmax=vmax_left,
             cmap=mymap, threshold=threshold, darkness = darkness,boundary=boundary, boundary_color=boundary_color)   
    
    gap = 150
    i1 = PNGWhiteTrim(os.path.join(save_path, 'fig.hcp.embed.L.0.png'))
    i2 = PNGWhiteTrim(os.path.join(save_path, 'fig.hcp.embed.L.1.png'))
    i4 = PNGWhiteTrim(os.path.join(save_path, 'fig.hcp.embed.R.0.png'))
    i3 = PNGWhiteTrim(os.path.join(save_path, 'fig.hcp.embed.R.1.png'))
    result = Image.new("RGBA", (gap*2+np.shape(i1)[1]+gap+np.shape(i2)[1]+gap+np.shape(i3)[1]+gap+np.shape(i4)[1] + int(gap*4.5), int(3*gap)+np.shape(i1)[0]+gap), (255, 255, 255, 255))
    result.paste(i2, (2*gap, int(3*gap)))
    result.paste(i1, (np.shape(i2)[1]+3*gap, int(3*gap)))
    result.paste(i3, (np.shape(i1)[1]+3*gap+np.shape(i2)[1]+gap + int(gap*0.5), int(3*gap)))
    result.paste(i4, (np.shape(i1)[1]+3*gap+np.shape(i2)[1]+gap+np.shape(i3)[1]+gap + int(gap*0.5), int(3*gap)))


    # Add the title and hemisphere labels.
    fig, ax = plt.subplots(figsize=(result.width / 90, 2.7)) 
    fig.patch.set_alpha(0)
    ax.text(0.5, 0.7, name, ha='center', va='center', fontsize=100)
    ax.text(0, 0, 'L', ha='center', va='center', fontsize=80)
    ax.text(1, 0, 'R', ha='center', va='center', fontsize=80)
    ax.axis('off')
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
    buf.seek(0)
    plt.close(fig)
    title_image = Image.open(buf)
    result.paste(title_image, (int(1.2*gap), gap))


    fig, ax = plt.subplots(figsize=(1, (result.height / 100) * 0.8))
    ax.axis('off')
    fig.patch.set_alpha(0)
    # Create a mappable object for the color bar.
    norm = plt.Normalize(vmin=vmin_left, vmax=vmax_left)
    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(mymap), norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=1, pad=0,aspect=10)
    cbar.ax.spines['top'].set_linewidth(2)
    cbar.ax.spines['right'].set_linewidth(2)
    cbar.ax.spines['bottom'].set_linewidth(2)
    cbar.ax.spines['left'].set_linewidth(2)

    # Set color-bar tick positions and labels.
    ticks = np.linspace(vmin_left, vmax_left, num=3)
    ticklabels = [f'{tick: .2f}' for tick in ticks]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(ticklabels)
    cbar.ax.tick_params(labelsize=60, length=30, width=3)

    # Render the Matplotlib canvas as a PIL image.
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
    buf.seek(0)
    colorbar_image = Image.open(buf)
    plt.close(fig)
    # Add the color bar to the combined surface image.
    result.paste(colorbar_image, (result.width - int(2.6 * gap), int(2.8 * gap)))
    result = result.convert("RGB")
    #result.save(os.path.join(save_path, name+'.png'))

    return result
