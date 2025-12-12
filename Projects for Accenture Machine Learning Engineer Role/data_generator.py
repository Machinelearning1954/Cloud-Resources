"""
Synthetic Army Vehicle Sensor Data Generator

Generates realistic sensor telemetry data for training predictive maintenance models.
Based on published specifications for military tactical vehicles.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import argparse
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VehicleSensorDataGenerator:
    """Generate synthetic sensor data for military vehicles."""
    
    def __init__(self, num_vehicles=1000, days=365, seed=42):
        """
        Initialize data generator.
        
        Args:
            num_vehicles: Number of vehicles to simulate
            days: Number of days of sensor data per vehicle
            seed: Random seed for reproducibility
        """
        self.num_vehicles = num_vehicles
        self.days = days
        self.seed = seed
        np.random.seed(seed)
        
        # Normal operating ranges for vehicle sensors
        self.normal_ranges = {
            'engine_temp': (180, 210),  # Fahrenheit
            'oil_pressure': (40, 60),   # PSI
            'coolant_level': (90, 100), # Percent
            'battery_voltage': (12.5, 14.5),  # Volts
            'fuel_consumption': (8, 15),  # MPG
            'vibration_level': (0.5, 2.0),  # G-force
            'transmission_temp': (150, 200),  # Fahrenheit
            'brake_pressure': (80, 120),  # PSI
            'tire_pressure_fl': (50, 60),  # PSI (front left)
            'tire_pressure_fr': (50, 60),  # PSI (front right)
            'tire_pressure_rl': (50, 60),  # PSI (rear left)
            'tire_pressure_rr': (50, 60),  # PSI (rear right)
        }
        
    def generate_vehicle_data(self, vehicle_id, failure_probability=0.15):
        """
        Generate sensor data for a single vehicle.
        
        Args:
            vehicle_id: Unique vehicle identifier
            failure_probability: Probability of component failure
            
        Returns:
            DataFrame with daily sensor readings
        """
        # Determine if this vehicle will have a failure
        will_fail = np.random.random() < failure_probability
        
        if will_fail:
            # Failure occurs randomly between day 30 and end of period
            failure_day = np.random.randint(30, self.days)
            failure_component = np.random.choice([
                'engine', 'transmission', 'brakes', 'cooling', 'electrical'
            ])
        else:
            failure_day = None
            failure_component = None
            
        data = []
        start_date = datetime(2024, 1, 1)
        
        # Initialize vehicle characteristics
        base_mileage = np.random.uniform(10000, 80000)
        daily_mileage = np.random.uniform(50, 200)
        last_maintenance_day = -np.random.randint(0, 180)  # Days since last maintenance
        
        for day in range(self.days):
            current_date = start_date + timedelta(days=day)
            current_mileage = base_mileage + (day * daily_mileage)
            days_since_maintenance = day - last_maintenance_day
            
            # Generate base sensor readings
            sensors = self._generate_normal_readings()
            
            # Add temporal trends and noise
            sensors = self._add_temporal_effects(sensors, day, self.days)
            
            # Add degradation based on usage
            sensors = self._add_usage_degradation(
                sensors, days_since_maintenance, current_mileage
            )
            
            # Add failure patterns if applicable
            if will_fail and day >= (failure_day - 14):
                days_to_failure = failure_day - day
                sensors = self._add_failure_patterns(
                    sensors, failure_component, days_to_failure
                )
            
            # Create record
            record = {
                'vehicle_id': f'V-{vehicle_id:05d}',
                'date': current_date,
                'day_index': day,
                'mileage': current_mileage,
                'days_since_maintenance': days_since_maintenance,
                'failure_occurred': 1 if (will_fail and day == failure_day) else 0,
                'failure_within_14_days': 1 if (will_fail and 0 <= (failure_day - day) <= 14) else 0,
                'failure_component': failure_component if will_fail else 'none',
                **sensors
            }
            
            data.append(record)
            
            # Simulate maintenance events
            if days_since_maintenance > 180 or (will_fail and day == failure_day):
                last_maintenance_day = day
                
        return pd.DataFrame(data)
    
    def _generate_normal_readings(self):
        """Generate sensor readings within normal operating ranges."""
        sensors = {}
        for sensor, (min_val, max_val) in self.normal_ranges.items():
            # Generate normally distributed values within range
            mean = (min_val + max_val) / 2
            std = (max_val - min_val) / 6  # 99.7% within range
            value = np.random.normal(mean, std)
            sensors[sensor] = np.clip(value, min_val, max_val)
        return sensors
    
    def _add_temporal_effects(self, sensors, day, total_days):
        """Add seasonal and temporal effects to sensor readings."""
        # Seasonal temperature variation (summer hotter)
        season_factor = np.sin(2 * np.pi * day / 365)
        sensors['engine_temp'] += season_factor * 10
        sensors['transmission_temp'] += season_factor * 8
        
        # Gradual battery degradation over time
        battery_degradation = (day / total_days) * 0.5
        sensors['battery_voltage'] -= battery_degradation
        
        return sensors
    
    def _add_usage_degradation(self, sensors, days_since_maintenance, mileage):
        """Add degradation patterns based on usage."""
        # Oil pressure decreases with time since maintenance
        oil_degradation = min(days_since_maintenance / 180, 1.0) * 10
        sensors['oil_pressure'] -= oil_degradation
        
        # Vibration increases with wear
        vibration_increase = (days_since_maintenance / 180) * 0.8
        sensors['vibration_level'] += vibration_increase
        
        # Fuel efficiency decreases with mileage
        fuel_degradation = (mileage / 100000) * 2
        sensors['fuel_consumption'] += fuel_degradation
        
        # Tire pressure slowly decreases
        tire_degradation = (days_since_maintenance / 180) * 3
        for tire in ['tire_pressure_fl', 'tire_pressure_fr', 
                     'tire_pressure_rl', 'tire_pressure_rr']:
            sensors[tire] -= tire_degradation * np.random.uniform(0.8, 1.2)
        
        return sensors
    
    def _add_failure_patterns(self, sensors, component, days_to_failure):
        """Add sensor patterns indicating impending failure."""
        # Severity increases as failure approaches
        severity = 1.0 - (days_to_failure / 14)
        
        if component == 'engine':
            sensors['engine_temp'] += severity * 30
            sensors['oil_pressure'] -= severity * 15
            sensors['vibration_level'] += severity * 2.5
            sensors['fuel_consumption'] += severity * 5
            
        elif component == 'transmission':
            sensors['transmission_temp'] += severity * 40
            sensors['vibration_level'] += severity * 3.0
            sensors['fuel_consumption'] += severity * 4
            
        elif component == 'brakes':
            sensors['brake_pressure'] -= severity * 25
            sensors['vibration_level'] += severity * 1.5
            
        elif component == 'cooling':
            sensors['engine_temp'] += severity * 35
            sensors['coolant_level'] -= severity * 20
            sensors['transmission_temp'] += severity * 25
            
        elif component == 'electrical':
            sensors['battery_voltage'] -= severity * 2.0
            # Add voltage fluctuations
            sensors['battery_voltage'] += np.random.normal(0, severity * 0.5)
        
        # Add random noise to make patterns less obvious
        for sensor in sensors:
            sensors[sensor] += np.random.normal(0, severity * 0.5)
            
        return sensors
    
    def generate_dataset(self, output_path=None):
        """
        Generate complete dataset for all vehicles.
        
        Args:
            output_path: Path to save CSV file (optional)
            
        Returns:
            DataFrame with all vehicle sensor data
        """
        logger.info(f"Generating data for {self.num_vehicles} vehicles over {self.days} days...")
        
        all_data = []
        for vehicle_id in range(self.num_vehicles):
            if vehicle_id % 100 == 0:
                logger.info(f"Processing vehicle {vehicle_id}/{self.num_vehicles}")
            
            vehicle_data = self.generate_vehicle_data(vehicle_id)
            all_data.append(vehicle_data)
        
        df = pd.concat(all_data, ignore_index=True)
        
        # Add derived features
        df = self._add_derived_features(df)
        
        logger.info(f"Generated {len(df)} records")
        logger.info(f"Failure rate: {df['failure_within_14_days'].mean():.2%}")
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            logger.info(f"Saved dataset to {output_path}")
        
        return df
    
    def _add_derived_features(self, df):
        """Add derived features for ML training."""
        # Sort by vehicle and date
        df = df.sort_values(['vehicle_id', 'date']).reset_index(drop=True)
        
        # Rolling statistics (7-day window)
        rolling_features = ['engine_temp', 'oil_pressure', 'vibration_level', 
                           'fuel_consumption', 'transmission_temp']
        
        for feature in rolling_features:
            df[f'{feature}_7d_mean'] = df.groupby('vehicle_id')[feature].transform(
                lambda x: x.rolling(7, min_periods=1).mean()
            )
            df[f'{feature}_7d_std'] = df.groupby('vehicle_id')[feature].transform(
                lambda x: x.rolling(7, min_periods=1).std().fillna(0)
            )
        
        # Lag features (previous day)
        for feature in rolling_features:
            df[f'{feature}_lag1'] = df.groupby('vehicle_id')[feature].shift(1).fillna(
                df[feature]
            )
        
        # Rate of change
        for feature in rolling_features:
            df[f'{feature}_change'] = df.groupby('vehicle_id')[feature].diff().fillna(0)
        
        # Tire pressure asymmetry
        df['tire_pressure_asymmetry'] = (
            df[['tire_pressure_fl', 'tire_pressure_fr', 
                'tire_pressure_rl', 'tire_pressure_rr']].std(axis=1)
        )
        
        return df


def main():
    """Command line interface for data generation."""
    parser = argparse.ArgumentParser(
        description='Generate synthetic Army vehicle sensor data'
    )
    parser.add_argument(
        '--num-vehicles', type=int, default=1000,
        help='Number of vehicles to simulate (default: 1000)'
    )
    parser.add_argument(
        '--days', type=int, default=365,
        help='Number of days of data per vehicle (default: 365)'
    )
    parser.add_argument(
        '--output', type=str, 
        default='data/synthetic/vehicle_sensor_data.csv',
        help='Output CSV file path'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Generate data
    generator = VehicleSensorDataGenerator(
        num_vehicles=args.num_vehicles,
        days=args.days,
        seed=args.seed
    )
    
    df = generator.generate_dataset(output_path=args.output)
    
    # Print summary statistics
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    print(f"Total records: {len(df):,}")
    print(f"Unique vehicles: {df['vehicle_id'].nunique():,}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"\nFailure statistics:")
    print(f"  Vehicles with failures: {df.groupby('vehicle_id')['failure_occurred'].max().sum():,}")
    print(f"  Total failure events: {df['failure_occurred'].sum():,}")
    print(f"  Records with failure risk (14-day window): {df['failure_within_14_days'].sum():,}")
    print(f"  Failure risk rate: {df['failure_within_14_days'].mean():.2%}")
    print(f"\nFailure by component:")
    print(df[df['failure_occurred'] == 1]['failure_component'].value_counts())
    print("="*60)


if __name__ == '__main__':
    main()
