import os
import logging
from typing import Dict
from datetime import datetime, timedelta

try:
    import copernicusmarine
except ImportError:
    copernicusmarine = None

logger = logging.getLogger(__name__)

class EnvironmentalService:
    """
    Handles fetching and caching of ocean current and wind data 
    (from CMEMS via copernicusmarine package) for the OpenDrift engine.
    """
    
    @staticmethod
    def get_forcing_data(incident_lat: float, incident_lon: float, timestamp_utc: str, force_cached: bool = False) -> Dict[str, str]:
        """
        Retrieves NetCDF forcing data or returns paths to cached files.
        Uses CMEMS live data if credentials are set, otherwise falls back to demo cache.
        """
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(base_dir, '..', 'data', 'environmental')
        os.makedirs(data_dir, exist_ok=True)
        
        cached_currents = os.path.join(data_dir, 'cached_currents.nc')
        cached_winds = os.path.join(data_dir, 'cached_winds.nc')
        
        cmems_user = os.getenv("COPERNICUS_USERNAME")
        cmems_pass = os.getenv("COPERNICUS_PASSWORD")
        
        # If forced to cached, or no credentials/library available, use cached mode
        if force_cached or not copernicusmarine or not cmems_user or not cmems_pass:
            return {
                "currents_path": cached_currents,
                "winds_path": cached_winds,
                "data_source_flag": "cached"
            }
            
        try:
            # We need data for 24 hours prior to the detection time
            end_time = datetime.fromisoformat(timestamp_utc.replace("Z", "+00:00"))
            start_time = end_time - timedelta(hours=36) # Buffer
            
            # Spatial bounding box for download (e.g. +/- 1.5 degrees from spill)
            min_lon, max_lon = incident_lon - 1.5, incident_lon + 1.5
            min_lat, max_lat = incident_lat - 1.5, incident_lat + 1.5
            
            currents_filename = f"cmems_currents_{int(end_time.timestamp())}.nc"
            currents_out = os.path.join(data_dir, currents_filename)
            
            if not os.path.exists(currents_out):
                # Download Global Ocean Physics Analysis and Forecast
                copernicusmarine.subset(
                    dataset_id="cmems_mod_glo_phy_anfc_0.083deg_PT1H-m",
                    variables=["uo", "vo"],
                    start_datetime=start_time.strftime("%Y-%m-%dT%H:%M:%S"),
                    end_datetime=end_time.strftime("%Y-%m-%dT%H:%M:%S"),
                    minimum_longitude=min_lon,
                    maximum_longitude=max_lon,
                    minimum_latitude=min_lat,
                    maximum_latitude=max_lat,
                    minimum_depth=0.49,
                    maximum_depth=0.51, # Surface currents only
                    output_filename=currents_out,
                    username=cmems_user,
                    password=cmems_pass,
                    force_download=True
                )
                
            return {
                "currents_path": currents_out,
                "winds_path": cached_winds, # Usually use ECMWF ERA5 or similar for winds, keeping cached for simplicity unless specified
                "data_source_flag": "live_cmems"
            }
            
        except Exception as e:
            logger.error(f"Live CMEMS download failed: {str(e)}. Falling back to cache.")
            return {
                "currents_path": cached_currents,
                "winds_path": cached_winds,
                "data_source_flag": "cached_fallback"
            }

    @staticmethod
    def get_synthetic_forcing_vector(lat: float, lon: float, timestamp: datetime) -> Dict[str, float]:
        """
        Provides a mathematical fallback generating synthetic but realistic
        ocean current and wind vectors for a given geographic region.
        """
        import math
        # Simple localized synthetic vector generation based on coordinate quadrants
        # In reality, this could sample a lightweight static grid.
        
        # Base current: ~0.5 m/s, varies by latitude (pseudo-Coriolis effect)
        base_u_current = 0.5 * math.cos(math.radians(lat))
        base_v_current = 0.3 * math.sin(math.radians(lon))
        
        # Wind: generally stronger, e.g., 5-10 m/s
        base_u_wind = 5.0 * math.cos(math.radians(lat + 45))
        base_v_wind = 5.0 * math.sin(math.radians(lon + 45))
        
        return {
            "u_current_ms": base_u_current,
            "v_current_ms": base_v_current,
            "u_wind_ms": base_u_wind,
            "v_wind_ms": base_v_wind
        }
        
    @staticmethod
    def calculate_kinematic_drift(
        start_lat: float, 
        start_lon: float, 
        start_time: datetime, 
        end_time: datetime,
        windage_coefficient: float = 0.03
    ) -> Dict:
        """
        Calculates dead-reckoning displacement based on time delta and synthetic forces.
        """
        vectors = EnvironmentalService.get_synthetic_forcing_vector(start_lat, start_lon, start_time)
        
        # Calculate time delta in seconds
        dt_seconds = (end_time - start_time).total_seconds()
        
        if dt_seconds == 0:
            return {
                "end_lat": start_lat,
                "end_lon": start_lon,
                "drift_vector": {**vectors, "drift_time_seconds": 0.0, "delta_x_m": 0.0, "delta_y_m": 0.0}
            }
            
        # Total drift velocity vector (Current + Windage * Wind)
        u_total = vectors["u_current_ms"] + (windage_coefficient * vectors["u_wind_ms"])
        v_total = vectors["v_current_ms"] + (windage_coefficient * vectors["v_wind_ms"])
        
        # Total displacement in meters
        dx_m = u_total * dt_seconds
        dy_m = v_total * dt_seconds
        
        # Rough conversion of meters displacement to decimal degrees
        # 1 degree lat ~ 111,111 meters
        # 1 degree lon ~ 111,111 * cos(lat) meters
        import math
        dlat = dy_m / 111111.0
        dlon = dx_m / (111111.0 * math.cos(math.radians(start_lat)))
        
        end_lat = start_lat + dlat
        end_lon = start_lon + dlon
        
        return {
            "end_lat": end_lat,
            "end_lon": end_lon,
            "drift_vector": {
                **vectors,
                "drift_time_seconds": dt_seconds,
                "delta_x_m": dx_m,
                "delta_y_m": dy_m
            }
        }
