import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def _to_jsonable(value):
    """Convert numpy/pandas types to JSON-friendly values."""
    if isinstance(value, (np.integer, np.int64)):
        return int(value)
    if isinstance(value, (np.floating, np.float64)):
        return float(value)
    if pd.isna(value):
        return None
    return value


class InsuranceCorrelationAnalyzer:

    DEFAULT_CATEGORICAL = [
        "MULTIPRODICTDISCOUTN_TP",
        "INSUREDITEM_TP",
        "POLICYRATEDSTATE_TP",
        "RATEDCOUNTY_TP",
        "INTEGRATEDCOVERAGE_TP",
        "PROPERTYCOVERAGESUBTYPE_TP",
        "CREDITMODEL_CD",
        "SEASONAL_IN",
        "CONSTRUCTION_TP",
        "ROOF_TP",
    ]

    DEFAULT_NUMERICAL = [
        "DIRECTWRITTENPREMIUNM_AM",
        "PPCVRGLIMIT_AM",
        "DWELLINGSQUAREFEET_CT",
        "DWELLINGSTORY_CT",
        "EARNEDEXPOSURE_CT",
        "EARNEDPREMIUM_AM",
        "WRITENEXPOSURE_CT",
        "TAX_AM",
        "POLICYTERM_CT",
    ]

    DEFAULT_DATE = ["POLICYEFFECTIVE_DT"]

    # Additional columns from your data
    ADDITIONAL_COLS = ["GROSSLOSSPAIO_AM", "ACCOUNTING_MONTH", "ZIP"]

    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.correlations = {}

    def analyze_all_correlations(self):
        """Run all correlation analyses."""
        print("Starting correlation analysis...")
        self._numerical_correlations()
        self._categorical_associations()
        self._numerical_categorical_relationships()
        print("Correlation analysis complete!")
        return self

    def _get_available_columns(self):
        """Get columns that exist in the dataframe."""
        df = self.data

        categorical = [c for c in self.DEFAULT_CATEGORICAL if c in df.columns]
        numeric = [c for c in self.DEFAULT_NUMERICAL if c in df.columns]
        date_cols = [c for c in self.DEFAULT_DATE if c in df.columns]

        # Add additional columns if they exist
        for col in self.ADDITIONAL_COLS:
            if col in df.columns:
                # Try to infer type
                if df[col].dtype in ['int64', 'float64']:
                    numeric.append(col)
                else:
                    categorical.append(col)

        return categorical, numeric, date_cols

    def _numerical_correlations(self):
        """Compute Pearson correlations between numerical columns."""
        print("Computing numerical correlations...")

        _, numeric_cols, _ = self._get_available_columns()

        if len(numeric_cols) < 2:
            print("Not enough numerical columns for correlation.")
            return

        # Get numeric data and convert to numeric
        numeric_data = self.data[numeric_cols].apply(pd.to_numeric, errors='coerce')

        # Compute correlation matrix
        corr_matrix = numeric_data.corr(method='pearson')

        # Store full matrix
        self.correlations['numerical_correlation_matrix'] = {
            'matrix': corr_matrix.to_dict(),
            'columns': numeric_cols
        }

        # Find strong correlations (|r| > 0.5)
        strong_correlations = []
        for i, col1 in enumerate(numeric_cols):
            for j, col2 in enumerate(numeric_cols):
                if i < j:  # Only upper triangle
                    corr_val = corr_matrix.loc[col1, col2]
                    if not pd.isna(corr_val) and abs(corr_val) > 0.5:
                        strong_correlations.append({
                            'column1': col1,
                            'column2': col2,
                            'correlation': float(corr_val),
                            'strength': 'strong' if abs(corr_val) > 0.7 else 'moderate'
                        })

        # Sort by absolute correlation
        strong_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
        self.correlations['strong_numerical_correlations'] = strong_correlations

        print(f"  Found {len(strong_correlations)} strong numerical correlations")

    def _categorical_associations(self):
        """Compute Cramér's V for categorical variables."""
        print("Computing categorical associations (Cramér's V)...")

        categorical_cols, _, _ = self._get_available_columns()

        if len(categorical_cols) < 2:
            print("Not enough categorical columns for association.")
            return

        associations = []

        for i, col1 in enumerate(categorical_cols):
            for j, col2 in enumerate(categorical_cols):
                if i < j:  # Only upper triangle
                    cramers_v = self._cramers_v(self.data[col1], self.data[col2])

                    if cramers_v is not None and cramers_v > 0.3:
                        associations.append({
                            'column1': col1,
                            'column2': col2,
                            'cramers_v': float(cramers_v),
                            'strength': 'strong' if cramers_v > 0.5 else 'moderate'
                        })

        # Sort by Cramér's V
        associations.sort(key=lambda x: x['cramers_v'], reverse=True)
        self.correlations['categorical_associations'] = associations

        print(f"  Found {len(associations)} moderate/strong categorical associations")

    def _cramers_v(self, x, y):
        """Calculate Cramér's V statistic for categorical-categorical association."""
        try:
            # Create contingency table
            confusion_matrix = pd.crosstab(x, y)

            # Chi-square test
            chi2 = 0
            n = confusion_matrix.sum().sum()

            # Expected frequencies
            row_sums = confusion_matrix.sum(axis=1)
            col_sums = confusion_matrix.sum(axis=0)

            for i in confusion_matrix.index:
                for j in confusion_matrix.columns:
                    observed = confusion_matrix.loc[i, j]
                    expected = (row_sums[i] * col_sums[j]) / n
                    if expected > 0:
                        chi2 += (observed - expected) ** 2 / expected

            # Cramér's V
            min_dim = min(len(confusion_matrix.index) - 1, len(confusion_matrix.columns) - 1)
            if min_dim > 0 and n > 0:
                cramers_v = np.sqrt(chi2 / (n * min_dim))
                return cramers_v

        except Exception:
            pass

        return None

    def _numerical_categorical_relationships(self):
        """Analyze correlation ratio (eta) between categorical and numerical variables."""
        print("Computing categorical-numerical relationships...")

        categorical_cols, numeric_cols, _ = self._get_available_columns()

        if not categorical_cols or not numeric_cols:
            print("Need both categorical and numerical columns.")
            return

        relationships = []

        for cat_col in categorical_cols:
            for num_col in numeric_cols:
                eta_squared = self._correlation_ratio(self.data[cat_col], self.data[num_col])

                if eta_squared is not None and eta_squared > 0.1:
                    relationships.append({
                        'categorical': cat_col,
                        'numerical': num_col,
                        'eta_squared': float(eta_squared),
                        'strength': 'strong' if eta_squared > 0.25 else 'moderate'
                    })

        # Sort by eta squared
        relationships.sort(key=lambda x: x['eta_squared'], reverse=True)
        self.correlations['categorical_numerical_relationships'] = relationships

        print(f"  Found {len(relationships)} moderate/strong cat-num relationships")

    def _correlation_ratio(self, categories, values):
        """Calculate correlation ratio (eta) for categorical-numerical relationship."""
        try:
            # Convert to numeric
            values_numeric = pd.to_numeric(values, errors='coerce')

            # Create a dataframe
            df = pd.DataFrame({'cat': categories, 'val': values_numeric})
            df = df.dropna()

            if len(df) == 0:
                return None

            # Overall mean
            mean_total = df['val'].mean()

            # Group means
            grouped = df.groupby('cat')['val']

            # Between-group variance
            ss_between = sum(
                len(group) * (group.mean() - mean_total) ** 2
                for _, group in grouped
            )

            # Total variance
            ss_total = sum((df['val'] - mean_total) ** 2)

            if ss_total > 0:
                eta_squared = ss_between / ss_total
                return eta_squared

        except Exception:
            pass

        return None

    def generate_correlation_report(self, output_path: str = "correlation_report.json"):
        """Save correlation analysis to JSON file."""
        report = {
            'meta': {
                'rows': int(self.data.shape[0]),
                'columns': int(self.data.shape[1]),
            },
            'correlations': self.correlations
        }

        output_path = Path(output_path)
        output_path.write_text(
            json.dumps(report, indent=2, default=_to_jsonable),
            encoding='utf-8'
        )
        print(f"\nCorrelation report saved to: {output_path}")
        return report

    def visualize_numerical_correlations(self, output_path: str = "correlation_heatmap.png"):
        """Create a heatmap of numerical correlations."""
        _, numeric_cols, _ = self._get_available_columns()

        if len(numeric_cols) < 2:
            print("Not enough numerical columns to visualize.")
            return

        numeric_data = self.data[numeric_cols].apply(pd.to_numeric, errors='coerce')
        corr_matrix = numeric_data.corr(method='pearson')

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8}
        )
        plt.title('Numerical Correlations Heatmap', fontsize=16, pad=20)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to: {output_path}")
        plt.close()

    def print_summary(self):
        """Print human-readable summary of correlations."""
        print("\n" + "=" * 80)
        print("CORRELATION ANALYSIS SUMMARY")
        print("=" * 80)

        # Numerical correlations
        if 'strong_numerical_correlations' in self.correlations:
            strong_num = self.correlations['strong_numerical_correlations']
            print(f"\nStrong Numerical Correlations (|r| > 0.5): {len(strong_num)}")
            for corr in strong_num[:5]:  # Top 5
                print(f"  {corr['column1']} ↔ {corr['column2']}: r = {corr['correlation']:.3f}")

        # Categorical associations
        if 'categorical_associations' in self.correlations:
            cat_assoc = self.correlations['categorical_associations']
            print(f"\nCategorical Associations (Cramér's V > 0.3): {len(cat_assoc)}")
            for assoc in cat_assoc[:5]:  # Top 5
                print(f"  {assoc['column1']} ↔ {assoc['column2']}: V = {assoc['cramers_v']:.3f}")

        # Cat-Num relationships
        if 'categorical_numerical_relationships' in self.correlations:
            cat_num = self.correlations['categorical_numerical_relationships']
            print(f"\nCategorical-Numerical Relationships (η² > 0.1): {len(cat_num)}")
            for rel in cat_num[:5]:  # Top 5
                print(f"  {rel['categorical']} → {rel['numerical']}: η² = {rel['eta_squared']:.3f}")

        print("=" * 80 + "\n")


# Configuration
DATA_PATH = "production_insurance_data.csv"  # Change to your file path
REPORT_PATH = "correlation_report.json"
HEATMAP_PATH = "correlation_heatmap.png"


if __name__ == "__main__":
    # Supports both CSV and XLSX
    print(f"Loading data from: {DATA_PATH}")

    if DATA_PATH.endswith('.xlsx'):
        data = pd.read_excel(DATA_PATH)
    else:
        data = pd.read_csv(DATA_PATH)

    print(f"Loaded {len(data):,} rows and {len(data.columns)} columns")

    # Run correlation analysis
    analyzer = InsuranceCorrelationAnalyzer(data)
    analyzer.analyze_all_correlations()
    analyzer.print_summary()

    # Generate outputs
    analyzer.generate_correlation_report(REPORT_PATH)
    analyzer.visualize_numerical_correlations(HEATMAP_PATH)

    print("\n✓ Done")
