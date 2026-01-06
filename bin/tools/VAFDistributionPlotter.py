import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class VAFDistributionPlotter:
    """
    A class to read, filter, and visualize Variant Allele Frequency (VAF) data
    from pileup files across autosomes (chr1-chr22) with equal-width chromosome bins.
    
    Attributes
    ----------
    pileup_file_path : str
        Path to the gzipped pileup TSV file.
    chr_to_highlight : list[str] or None
        List of chromosome names to highlight in red, e.g., ['chr1', 'chr2'].
    depth_filter : int
        Minimum depth threshold for filtering.
    list_filter : list or None
        List of chr_pos strings to filter the data.
    title : str or None
        Title for the plot.
    alt_col : str
        Column name for alternate read counts.
    dp_col : str
        Column name for depth.
    filter_heterozygous : bool
        Whether to filter for specific VAF ranges (heterozygous sites).
    filter_snp_type : bool
        Whether to filter for specific SNP types (A/T variants).
    autosomes : list[str]
        List of autosomal chromosome names (chr1-chr22).
    df : pd.DataFrame or None
        The filtered dataframe with computed VAF and plotting coordinates.
    """
    
    def __init__(
        self,
        pileup_file_path,
        chr_to_highlight=None,
        depth_filter=60,
        list_filter=None,
        title=None,
        alt_col='cfDNA_alt_reads',
        dp_col='current_depth',
        filter_heterozygous=False,
        filter_snp_type=False
    ):
        """
        Initialize the VAFDistributionPlotter with configuration parameters.
        
        Parameters
        ----------
        pileup_file_path : str
            Path to the gzipped pileup TSV file.
        chr_to_highlight : list[str] or None, optional
            List of chromosome names to highlight. Default is None.
        depth_filter : int, optional
            Minimum depth for filtering. Default is 60.
        list_filter : list or None, optional
            List of chr_pos strings to filter. Default is None.
        title : str or None, optional
            Plot title. Default is None.
        alt_col : str, optional
            Column name for alternate reads. Default is 'cfDNA_alt_reads'.
        dp_col : str, optional
            Column name for depth. Default is 'current_depth'.
        filter_heterozygous : bool, optional
            Apply VAF filtering for heterozygous sites. Default is False.
        filter_snp_type : bool, optional
            Filter for A/T SNP types only. Default is False.
        """
        self.pileup_file_path = pileup_file_path
        self.chr_to_highlight = chr_to_highlight if chr_to_highlight is not None else []
        self.depth_filter = depth_filter
        self.list_filter = list_filter
        self.title = title
        self.alt_col = alt_col
        self.dp_col = dp_col
        self.filter_heterozygous = filter_heterozygous
        self.filter_snp_type = filter_snp_type
        self.autosomes = [f'chr{i}' for i in range(1, 23)]
        self.df = None
    
    def load_data(self):
        """
        Load the gzipped pileup TSV file into a pandas DataFrame.
        
        Returns
        -------
        pd.DataFrame
            The loaded dataframe.
        """
        self.df = pd.read_csv(self.pileup_file_path, sep='\t', compression='gzip')
        return self.df
    
    def apply_depth_filter(self):
        """
        Filter data based on minimum depth threshold.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        self.df = self.df[self.df[self.dp_col] > self.depth_filter].copy()
    
    def apply_list_filter(self):
        """
        Filter data based on a list of chr_pos strings.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        if self.list_filter is not None:
            self.df['chr_pos'] = self.df['chr'].astype(str) + '_' + self.df['pos'].astype(str)
            self.df = self.df[self.df['chr_pos'].isin(self.list_filter)]
    
    def compute_vaf(self):
        """
        Compute Variant Allele Frequency (VAF) as alt_reads / depth.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        self.df['vaf'] = self.df[self.alt_col] / self.df[self.dp_col]
    
    def apply_vaf_filter(self):
        """
        Apply VAF-based filtering for heterozygous sites if enabled.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        if self.filter_heterozygous:
            # Filter for homozygous reference-like sites (VAF: 0.8-0.99)
            self.df = self.df[((self.df['vaf'] > 0.8) & (self.df['vaf'] < 0.99))]
    
    def apply_snp_type_filter(self):
        """
        Filter for specific SNP types (A/T variants) if enabled.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        if self.filter_snp_type:
            self.df = self.df[(self.df['ref'].isin(['A', 'T'])) & 
                             (self.df['alt'].isin(['A', 'T']))]
    
    def filter_autosomes(self):
        """
        Keep only autosomal chromosomes (chr1-chr22).
        
        Returns
        -------
        bool
            True if data remains after filtering, False otherwise.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        self.df = self.df[self.df['chr'].isin(self.autosomes)].copy()
        
        if self.df.empty:
            print("No data after filtering for autosomes and depth.")
            return False
        return True
    
    def prepare_plot_coordinates(self):
        """
        Prepare x-axis coordinates for equal-width chromosome bins and
        assign colors based on highlighted chromosomes.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Index map for equal-width bins: each chromosome occupies [i, i+1) on x-axis
        chr_index = {c: i for i, c in enumerate(self.autosomes)}
        self.df['chr_idx'] = self.df['chr'].map(chr_index)
        
        # For within-chromosome relative position, normalize by observed span per chr
        # (If a chr has only a single position, avoid div-by-zero by using span=1)
        bounds = self.df.groupby('chr')['pos'].agg(min_pos='min', max_pos='max')
        self.df = self.df.merge(bounds, left_on='chr', right_index=True, how='left')
        span = (self.df['max_pos'] - self.df['min_pos'])
        span = span.where(span > 0, 1)  # replace 0 with 1 safely
        self.df['within_chr_x'] = (self.df['pos'] - self.df['min_pos']) / span
        
        # Build equal-width genome-like x: integer bin + normalized within-chr position
        self.df['genome_x'] = self.df['chr_idx'].astype(float) + self.df['within_chr_x']
        
        # Determine colors: highlighted chromosomes in red, others default color
        mask_hl = self.df['chr'].isin(self.chr_to_highlight)
        self.df['color'] = np.where(mask_hl, 'red', 'tab:blue')
    
    def create_plot(self):
        """
        Create and display the VAF scatter plot with chromosome boundaries.
        
        Returns
        -------
        tuple
            A tuple containing (fig, ax) matplotlib objects.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(16, 6))
        
        # Set title with average depth information
        mean_depth = self.df[self.dp_col].mean()
        ax.set_title(
            f'Variant Allele Frequency (VAF), {self.title}\nSNP site mean depth : {mean_depth:.1f}',
            fontsize=14,
            fontweight='bold'
        )
        
        # Scatter plot
        ax.scatter(self.df['genome_x'], self.df['vaf'], s=6, alpha=0.6, c=self.df['color'])
        
        # Axis formatting
        ax.set_xlim(0, 22)  # 22 equal-width bins
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel('Chromosomes')
        ax.set_ylabel('VAF')
        
        # Draw black solid lines to delineate chromosome boundaries (between bins)
        for b in range(1, 22):
            ax.axvline(b, color='black', linewidth=1.0)
        
        # Put chromosome labels at bin centers
        ax.set_xticks(np.arange(0.5, 22.5, 1.0))
        ax.set_xticklabels(self.autosomes, rotation=0)
        
        # Light horizontal grid for readability
        ax.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig, ax
    
    def get_chr_pos_list(self):
        """
        Get a list of chromosome positions from the filtered data.
        
        Returns
        -------
        list
            List of chr_pos strings.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        if 'chr_pos' not in self.df.columns:
            self.df['chr_pos'] = self.df['chr'].astype(str) + '_' + self.df['pos'].astype(str)
        
        return list(self.df['chr_pos'])
    
    def plot(self):
        """
        Main method to execute the complete pipeline: load, filter, compute VAF, and plot.
        
        Returns
        -------
        tuple
            A tuple containing (list of chr_pos, filtered dataframe).
        """
        # Load data
        self.load_data()
        
        # Apply filters
        self.apply_depth_filter()
        self.apply_list_filter()
        
        # Compute VAF
        self.compute_vaf()
        
        # Apply VAF and SNP type filters
        self.apply_vaf_filter()
        self.apply_snp_type_filter()
        
        # Filter for autosomes
        if not self.filter_autosomes():
            return None, None
        
        # Prepare plot data
        self.prepare_plot_coordinates()
        
        # Create plot
        self.create_plot()
        
        # Get chr_pos list
        chr_pos_list = self.get_chr_pos_list()
        
        return chr_pos_list, self.df


# Legacy function wrapper for backward compatibility
def vaf_scatter_plot(
    pileup_file_path,
    chr_to_highlight=None,
    depth_filter=60,
    list_filter=None,
    title=None,
    alt_col='cfDNA_alt_reads',
    dp_col='current_depth',
    filter_heterozygous=False,
    filter_snp_type=False
):
    """
    Legacy function wrapper for backward compatibility.
    
    Read a gzipped pileup TSV, compute VAF, and plot all autosomes (chr1–chr22)
    on a single figure with equal-width chromosome bins.
    
    Changes vs. previous version:
      1) Force equal width for each chromosome on the x-axis (chr1..chr22).
      2) Allow highlighting specific chromosomes via 'chr_to_highlight' list; points
         on those chromosomes are colored red.
      3) y-limits set to [-0.05, 1.05].
      4) Figure size set to (16, 6).

    Parameters
    ----------
    pileup_file_path : str
        Path to the gzipped pileup TSV file.
    chr_to_highlight : list[str] or None
        List of chromosome names to highlight, e.g., ['chr1', 'chr2'].
        Default is None (no highlight).
    depth_filter : int
        Minimum depth threshold for filtering.
    list_filter : list or None
        List of chr_pos strings to filter the data.
    title : str or None
        Title for the plot.
    alt_col : str
        Column name for alternate read counts.
    dp_col : str
        Column name for depth.
    filter_heterozygous : bool
        Whether to filter for specific VAF ranges.
    filter_snp_type : bool
        Whether to filter for specific SNP types.
    
    Returns
    -------
    tuple
        A tuple containing (list of chr_pos, filtered dataframe).
    """
    plotter = VAFDistributionPlotter(
        pileup_file_path=pileup_file_path,
        chr_to_highlight=chr_to_highlight,
        depth_filter=depth_filter,
        list_filter=list_filter,
        title=title,
        alt_col=alt_col,
        dp_col=dp_col,
        filter_heterozygous=filter_heterozygous,
        filter_snp_type=filter_snp_type
    )
    
    return plotter.plot()