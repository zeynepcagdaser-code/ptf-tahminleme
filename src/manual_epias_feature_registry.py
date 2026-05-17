from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import PROJECT_ROOT


REGISTRY_PATH = PROJECT_ROOT / "data" / "processed" / "external_features" / "manual_selected_endpoint_registry.csv"


@dataclass(frozen=True)
class FeatureEndpoint:
    feature_name: str
    market_type: str
    service: str
    endpoint_path: str
    method: str = "POST"
    frequency: str = "hourly"
    requires_date_range: bool = True
    sort_field: str = "date"
    date_mode: str = "range"
    extra_body: dict[str, Any] | None = None
    notes: str = ""


SELECTED_ENDPOINTS: list[FeatureEndpoint] = [
    FeatureEndpoint("ptf", "electricity_dam", "electricity", "/v1/markets/dam/data/mcp", frequency="hourly"),
    FeatureEndpoint("kesinlesmemis_ptf", "electricity_dam", "electricity", "/v1/markets/dam/data/interim-mcp", frequency="hourly"),
    FeatureEndpoint("gop_islem_hacmi", "electricity_dam", "electricity", "/v1/markets/dam/data/day-ahead-market-trade-volume", frequency="hourly"),
    FeatureEndpoint("gop_fiyattan_bagimsiz_alis", "electricity_dam", "electricity", "/v1/markets/dam/data/price-independent-bid", frequency="hourly"),
    FeatureEndpoint("gop_fiyattan_bagimsiz_satis", "electricity_dam", "electricity", "/v1/markets/dam/data/price-independent-offer", frequency="hourly"),
    FeatureEndpoint("gop_eslesme_miktari", "electricity_dam", "electricity", "/v1/markets/dam/data/clearing-quantity", frequency="hourly"),
    FeatureEndpoint("gop_blok_alis_miktari", "electricity_dam", "electricity", "/v1/markets/dam/data/amount-of-block-buying", frequency="hourly"),
    FeatureEndpoint("gop_blok_satis_miktari", "electricity_dam", "electricity", "/v1/markets/dam/data/amount-of-block-selling", frequency="hourly"),
    FeatureEndpoint("gop_teklif_edilen_alis", "electricity_dam", "electricity", "/v1/markets/dam/data/submitted-bid-order-volume", frequency="hourly"),
    FeatureEndpoint("gop_teklif_edilen_satis", "electricity_dam", "electricity", "/v1/markets/dam/data/submitted-sales-order-volume", frequency="hourly"),
    FeatureEndpoint("gop_arz_talep", "electricity_dam", "electricity", "/v1/markets/dam/data/supply-demand", frequency="hourly"),
    FeatureEndpoint("gip_agirlikli_ortalama_fiyat", "electricity_idm", "electricity", "/v1/markets/idm/data/weighted-average-price", frequency="hourly"),
    FeatureEndpoint("gip_islem_hacmi", "electricity_idm", "electricity", "/v1/markets/idm/data/trade-value", frequency="hourly"),
    FeatureEndpoint("gip_eslesme_miktari", "electricity_idm", "electricity", "/v1/markets/idm/data/matching-quantity", frequency="hourly"),
    FeatureEndpoint("smf", "electricity_bpm", "electricity", "/v1/markets/bpm/data/system-marginal-price", frequency="hourly"),
    FeatureEndpoint("sistem_yonu", "electricity_bpm", "electricity", "/v1/markets/bpm/data/system-direction", frequency="hourly"),
    FeatureEndpoint("yal_talimat_miktari", "electricity_bpm", "electricity", "/v1/markets/bpm/data/order-summary-up", frequency="hourly"),
    FeatureEndpoint("yat_talimat_miktari", "electricity_bpm", "electricity", "/v1/markets/bpm/data/order-summary-down", frequency="hourly"),
    FeatureEndpoint("dengesizlik_miktari", "electricity_imbalance", "electricity", "/v1/markets/imbalance/data/imbalance-quantity", frequency="monthly"),
    FeatureEndpoint("dengesizlik_tutari", "electricity_imbalance", "electricity", "/v1/markets/imbalance/data/imbalance-amount", frequency="monthly"),
    FeatureEndpoint("gercek_zamanli_tuketim", "electricity_consumption", "electricity", "/v1/consumption/data/realtime-consumption", frequency="hourly"),
    FeatureEndpoint("yuk_tahmin_plani", "electricity_consumption", "electricity", "/v1/consumption/data/load-estimation-plan", frequency="hourly"),
    FeatureEndpoint("talep_tahmini", "electricity_consumption", "electricity", "/v1/consumption/data/demand-forecast", frequency="yearly", requires_date_range=False, sort_field="year", date_mode="none"),
    FeatureEndpoint("gercek_zamanli_uretim", "electricity_generation", "electricity", "/v1/generation/data/realtime-generation", frequency="hourly"),
    FeatureEndpoint("uretim_tahmini", "electricity_renewables", "electricity", "/v1/renewables/data/generation-forecast", frequency="hourly"),
    FeatureEndpoint("kgup", "electricity_generation", "electricity", "/v1/generation/data/dpp", frequency="hourly", notes="KGUP endpoint path resmi dokumanda dpp olarak gecebilir; hata loglanir."),
    FeatureEndpoint("eak", "electricity_generation", "electricity", "/v1/generation/data/aic", frequency="hourly", notes="Emre amade kapasite."),
    FeatureEndpoint("uevm", "electricity_generation", "electricity", "/v1/generation/data/injection-quantity", frequency="monthly"),
    FeatureEndpoint("yekdem_gercek_zamanli_uretim", "electricity_renewables", "electricity", "/v1/renewables/data/licensed-realtime-generation", frequency="hourly"),
    FeatureEndpoint("res_uretim_ve_tahmin", "electricity_renewables", "electricity", "/v1/renewables/data/res-generation-and-forecast", frequency="hourly"),
    FeatureEndpoint("yek_bedeli", "electricity_renewables", "electricity", "/v1/renewables/data/licensed-generation-cost", frequency="monthly"),
    FeatureEndpoint("yekdem_portfoy_geliri", "electricity_renewables", "electricity", "/v1/renewables/data/portfolio-income", frequency="monthly"),
    FeatureEndpoint("baraj_aktif_doluluk", "electricity_dams", "electricity", "/v1/dams/data/active-fullness", frequency="daily"),
    FeatureEndpoint("baraj_aktif_hacim", "electricity_dams", "electricity", "/v1/dams/data/active-volume", frequency="daily"),
    FeatureEndpoint("baraj_gunluk_hacim", "electricity_dams", "electricity", "/v1/dams/data/daily-volume", frequency="daily"),
    FeatureEndpoint("baraj_gunluk_kot", "electricity_dams", "electricity", "/v1/dams/data/daily-kot", frequency="daily"),
    FeatureEndpoint("pfk_fiyat", "electricity_ancillary", "electricity", "/v1/markets/ancillary-services/data/primary-frequency-capacity-price", frequency="hourly"),
    FeatureEndpoint("primer_frekans_rezerv_miktari", "electricity_ancillary", "electricity", "/v1/markets/ancillary-services/data/primary-frequency-capacity-amount", frequency="hourly"),
    FeatureEndpoint("sfk_fiyat", "electricity_ancillary", "electricity", "/v1/markets/ancillary-services/data/secondary-frequency-capacity-price", frequency="hourly"),
    FeatureEndpoint("sekonder_frekans_rezerv_miktari", "electricity_ancillary", "electricity", "/v1/markets/ancillary-services/data/secondary-frequency-capacity-amount", frequency="hourly"),
    FeatureEndpoint("ia_alis_miktari", "electricity_bilateral", "electricity", "/v1/markets/bilateral-contracts/data/bilateral-contracts-bid-quantity", frequency="monthly"),
    FeatureEndpoint("ia_satis_miktari", "electricity_bilateral", "electricity", "/v1/markets/bilateral-contracts/data/bilateral-contracts-offer-quantity", frequency="monthly"),
    FeatureEndpoint("enterkonneksiyon_kapasite", "electricity_transmission", "electricity", "/v1/transmission/data/nominal-capacity", frequency="daily"),
    FeatureEndpoint("hat_kapasiteleri", "electricity_transmission", "electricity", "/v1/transmission/data/line-capacities", frequency="daily"),
    FeatureEndpoint("iskk", "electricity_transmission", "electricity", "/v1/transmission/data/iskk-list", frequency="monthly"),
    FeatureEndpoint("kisit_maliyeti", "electricity_transmission", "electricity", "/v1/transmission/data/congestion-cost", frequency="monthly", extra_body={"orderType": "BOTH_REGULATIONS", "priceType": "MCP"}),
    FeatureEndpoint("spot_gaz_referans_fiyati", "natural_gas_sgp", "natural_gas", "/v1/markets/sgp/data/daily-reference-price", frequency="daily", sort_field="gasDay"),
    FeatureEndpoint("spot_gaz_islem_hacmi", "natural_gas_sgp", "natural_gas", "/v1/markets/sgp/data/total-trade-volume", frequency="daily", sort_field="gasDay"),
    FeatureEndpoint("spot_gaz_eslesme_miktari", "natural_gas_sgp", "natural_gas", "/v1/markets/sgp/data/match-quantity", frequency="daily", sort_field="gasDay"),
    FeatureEndpoint("spot_gaz_agirlikli_ortalama_fiyat", "natural_gas_sgp", "natural_gas", "/v1/markets/sgp/data/sgp-price", frequency="daily", sort_field="gasDay"),
    FeatureEndpoint("dogal_gaz_sistem_yonu", "natural_gas_sgp", "natural_gas", "/v1/markets/sgp/data/system-direction", frequency="daily", sort_field="gasDay"),
    FeatureEndpoint("dogal_gaz_dengesizlik", "natural_gas_sgp", "natural_gas", "/v1/markets/sgp/data/imbalance-amount", frequency="daily", sort_field="gasDay", date_mode="period"),
    FeatureEndpoint("dogal_gaz_talep", "natural_gas_transmission", "natural_gas", "/v1/transmission/data/day-ahead", frequency="daily"),
    FeatureEndpoint("dogal_gaz_tuketim", "natural_gas_transmission", "natural_gas", "/v1/transmission/data/daily-actualization-amount", frequency="daily"),
    FeatureEndpoint("dogal_gaz_arz", "natural_gas_sgp", "natural_gas", "/v1/markets/sgp/data/physical-realization", frequency="daily", sort_field="gasDay"),
    FeatureEndpoint("dogal_gaz_giris", "natural_gas_transmission", "natural_gas", "/v1/transmission/data/realization-entry-amount", frequency="daily"),
    FeatureEndpoint("dogal_gaz_cikis", "natural_gas_transmission", "natural_gas", "/v1/transmission/data/realization-exit-amount", frequency="daily"),
    FeatureEndpoint("lng_verileri", "natural_gas_transmission", "natural_gas", "/v1/transmission/data/entry-nomination", frequency="daily", notes="LNG ozel endpoint dokumanda ayri gorunmezse giris bildirimi proxy olarak denenir."),
    FeatureEndpoint("depolama_verileri", "natural_gas_transmission", "natural_gas", "/v1/transmission/data/stock-amount", frequency="daily"),
]


def get_selected_feature_registry() -> pd.DataFrame:
    return pd.DataFrame([asdict(item) for item in SELECTED_ENDPOINTS])


def save_selected_feature_registry(path: Path = REGISTRY_PATH) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    registry = get_selected_feature_registry()
    registry.to_csv(path, index=False)
    return registry
