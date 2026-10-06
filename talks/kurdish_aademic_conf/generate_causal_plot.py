"""
generate_causal_plot.py

Generates the Causal Effect Sizes Forest Plot for the WGSSS Impact Prize Poster.
Maps technical HRG codes to clinically intuitive, patient-understandable diagnostic descriptions
using tables/ICD_HRG_mapping.csv, tables/caus_hrg_lgb.csv, and tables/diagnosis_order.csv.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def generate_plot(output_path_png='images/causal_forest_plot.png', output_path_pdf='images/causal_forest_plot.pdf', return_fig=False):
    # Load data from local poster tables
    caus_df = pd.read_csv('tables/caus_hrg_lgb.csv')
    diag_order = pd.read_csv('tables/diagnosis_order.csv')

    df = pd.merge(caus_df, diag_order, on='profile', how='left')

    # Clinically intuitive mapping grounded in ICD_HRG_mapping.csv descriptions
    clinical_names = {
        'HE11': 'Hip & Femur Fractures',
        'HE': 'Musculoskeletal Trauma',
        'DZ11': 'Severe Pneumonia & Lung Infection',
        'EB10': 'Heart Attack & Cardiac Shock',
        'WJ': 'Sepsis & Severe Systemic Infection',
        'HC': 'Spinal & Joint Disorders',
        'AA': 'Stroke & Acute Neurological',
        'unknown': '★ Diagnostic Ambiguity (Uncoded Diagnosis)',
        'HD': 'Bone & Soft-Tissue Musculoskeletal',
        'FD11': 'Digestive & Oesophageal Cancers',
        'AA26': 'Cerebrovascular & Nerve Disorders',
        'LA04': 'Severe Kidney Infection / Pyelonephritis',
        'YQ50': 'Arterial Thrombosis & Vascular Disease',
        'EB14': 'Cardiomyopathy & Heart Disease',
        'EB': 'Valvular & Inflammatory Heart Disease',
        'DZ': 'Chronic Respiratory Conditions',
        'KC': 'Metabolic & Electrolyte Disorders',
        'LA': 'Kidney & Urinary Conditions',
        'EB07': 'Cardiac Arrhythmias & Conduction',
        'JD07': 'Skin & Subcutaneous Conditions',
        'GC17': 'Liver & Pancreatic Conditions',
        'WH': 'General Clinical Surveillance',
        'LB38': 'Urinary Symptoms & Haematuria',
        'MB': 'Gynaecological Conditions',
        'CB': 'Oral & Upper Respiratory',
        'FD': 'Gastrointestinal Infections',
        'LB': 'Bladder & Kidney Disorders',
        'FD10': 'General Intestinal Disorders',
        'FD02': 'Inflammatory Bowel Disease',
        'DZ65': 'Chronic Bronchitis & COPD',
        'FD03': 'Peptic Ulcer & GI Bleeding',
        'SA04': 'Iron Deficiency Anaemia',
        'DZ19': 'Wheezing & Bronchospasm',
        'EB12': 'Non-cardiac Chest Pain',
        'FD05': 'Non-specific Abdominal Pain',
        'WH52': 'Post-Surgical Cancer Surveillance'
    }

    df['clinical_name'] = df['profile'].map(clinical_names).fillna(df['profile'])

    # Sort by ate_days ascending so largest delays appear at top of Y-axis
    df = df.sort_values(by='ate_days', ascending=True).reset_index(drop=True)

    # Color definitions
    colors = []
    for idx, row in df.iterrows():
        if row['profile'] == 'unknown':
            colors.append('#dc2626') # Vivid crimson for unknown
        elif row['ci_low'] > 0:
            colors.append('#e05638') # Coral red for statistically significant positive
        elif row['ci_high'] < 0:
            colors.append('#0d9488') # Teal for statistically significant negative
        else:
            colors.append('#94a3b8') # Slate gray for not significant

    # Bubble sizing proportional to admission volume in NHS cohort
    prop_pct = df['proportion'] * 100
    sizes = 75 + (prop_pct / prop_pct.max()) * 520

    # Layout dimensions optimized for presentation slides - maximize width
    fig, ax = plt.subplots(figsize=(18, 9.5), dpi=100)
    fig.patch.set_facecolor('#FBFAF4')
    ax.set_facecolor('#FBFAF4')
    
    # Adjust margins to use full space
    plt.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.12)

    y_pos = np.arange(len(df))

    # Subtle background gridlines
    for y in y_pos:
        ax.axhline(y, color='#e2e8f0', linestyle='-', linewidth=0.8, zorder=1)

    # Solid vertical zero line
    ax.axvline(0, color='#475569', linestyle='-', linewidth=2.0, zorder=2)

    # 95% Confidence Interval error bars
    xerr_low = df['ate_days'] - df['ci_low']
    xerr_high = df['ci_high'] - df['ate_days']

    ax.errorbar(
        df['ate_days'], y_pos,
        xerr=[xerr_low, xerr_high],
        fmt='none',
        ecolor='#334155',
        elinewidth=1.5,
        capsize=3.0,
        capthick=1.4,
        zorder=3
    )

    # Scatter points (bubbles)
    ax.scatter(
        df['ate_days'], y_pos,
        s=sizes * 0.70,
        c=colors,
        edgecolors='#0f172a',
        linewidth=1.1,
        alpha=0.94,
        zorder=4
    )

    # Soft red highlight band behind the Unknown/Uncoded diagnosis row
    unknown_idx = df[df['profile'] == 'unknown'].index[0]
    ax.axhspan(unknown_idx - 0.48, unknown_idx + 0.48, color='#fee2e2', alpha=0.65, zorder=0)

    # Y-axis labels with font size balanced for slide readability
    ax.set_yticks(y_pos)
    labels = df['clinical_name'].tolist()
    ax.set_yticklabels(labels, fontsize=13, fontweight='medium', color='#0f172a')

    # Distinctive styling for Unknown Diagnosis label
    ax.get_yticklabels()[unknown_idx].set_fontweight('bold')
    ax.get_yticklabels()[unknown_idx].set_color('#b91c1c')
    ax.get_yticklabels()[unknown_idx].set_fontsize(14)

    # X-axis configuration
    ax.set_xlim(-8.5, 16.8)
    ax.set_xlabel('Causal Effect on Hospital Stay (Average Treatment Effect, Days)', fontsize=15, fontweight='bold', color='#0a2540', labelpad=10)
    ax.tick_params(axis='x', labelsize=13, colors='#1e293b')

    # Direction indicators
    ax.text(-4.5, 0.985, '← Shorter Hospital Stay',
            transform=ax.get_xaxis_transform(), ha='center', va='top',
            fontsize=13, fontweight='bold', color='#0d9488', clip_on=True)
    ax.text(8.0, 0.985, 'Extends Hospital Stay (Days) →',
            transform=ax.get_xaxis_transform(), ha='center', va='top',
            fontsize=13, fontweight='bold', color='#e05638', clip_on=True)

    # Header Title
    ax.set_title('Causal Effect of Diagnoses on Inpatient Length of Stay (LoS)', fontsize=16, fontweight='bold', color='#0a2540', pad=22)
    ax.text(0.5, 1.012, 'Bubble size = Volume (% of hospital admissions) | Error bars = 95% Confidence Interval (DML-IRM)',
            transform=ax.transAxes, ha='center', fontsize=12, color='#475569', style='italic')

    # Informative callout for Diagnostic Ambiguity
    ax.annotate('Diagnostic Ambiguity:\n+2.53 days causal delay\n(28.0% of all admissions)',
                xy=(df.loc[unknown_idx, 'ate_days'], unknown_idx),
                xytext=(df.loc[unknown_idx, 'ate_days'] + 3.8, unknown_idx - 3.8),
                fontsize=11, fontweight='bold', color='#b91c1c',
                arrowprops=dict(arrowstyle='->', color='#b91c1c', lw=2, connectionstyle='arc3,rad=-0.2'),
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#ffffff', edgecolor='#ef4444', lw=1.5),
                zorder=5)

    # Informative callout for HE11
    he11_idx = df[df['profile'] == 'HE11'].index[0]
    ax.annotate('Hip/Femur Fractures:\n+14.09 days delay',
                xy=(df.loc[he11_idx, 'ate_days'], he11_idx),
                xytext=(df.loc[he11_idx, 'ate_days'] - 5.8, he11_idx - 1.8),
                fontsize=11, fontweight='bold', color='#e05638',
                arrowprops=dict(arrowstyle='->', color='#e05638', lw=1.5, connectionstyle='arc3,rad=0.15'),
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#ffffff', edgecolor='#e05638', lw=1.3),
                zorder=5)

    # Frame lines
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#20808D')
        spine.set_linewidth(1.2)

    fig.savefig(output_path_png, dpi=300, bbox_inches='tight', facecolor='#FBFAF4')
    fig.savefig(output_path_pdf, bbox_inches='tight', facecolor='#FBFAF4')
    if return_fig:
        return fig
    plt.close(fig)
    print(f'Successfully generated: {output_path_png}')

if __name__ == '__main__':
    generate_plot()
