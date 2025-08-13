#!/bin/bash

task="train_feature_extractor"

prefix_task=" task="
prefix_dataset_name=" dataset_name="
prefix_feature_extractor=" train_feature_extractor.estimator="

estimators=("LITE")

dataset_names=("Adiac" "ArrowHead" "Beef" "BeetleFly" "BirdChicken" "BME" "Car" "CBF" "Chinatown" "ChlorineConcentration" "CinCECGTorso" "Coffee" "Computers" "CricketX" "CricketY" "CricketZ" "Crop" "DistalPhalanxOutlineCorrect" "DistalPhalanxOutlineAgeGroup" "DistalPhalanxTW" "Earthquakes" "ECG200" "ECG5000" "ECGFiveDays" "ElectricDevices" "EOGHorizontalSignal" "EOGVerticalSignal" "EthanolLevel" "FaceAll" "FaceFour" "FacesUCR" "Fish" "FordA" "FordB" "FreezerRegularTrain" "FreezerSmallTrain" "GunPoint" "GunPointAgeSpan" "GunPointMaleVersusFemale" "GunPointOldVersusYoung" "Ham" "Haptics" "Herring" "HouseTwenty" "InlineSkate" "InsectEPGRegularTrain" "InsectEPGSmallTrain" "InsectWingbeatSound" "ItalyPowerDemand" "LargeKitchenAppliances" "Lightning2" "Lightning7" "Mallat" "Meat" "MedicalImages" "MiddlePhalanxOutlineCorrect" "MiddlePhalanxOutlineAgeGroup" "MiddlePhalanxTW" "MixedShapesRegularTrain" "MixedShapesSmallTrain" "MoteStrain" "NonInvasiveFetalECGThorax1" "NonInvasiveFetalECGThorax2" "OliveOil" "OSULeaf" "PhalangesOutlinesCorrect" "PigAirwayPressure" "PigArtPressure" "PigCVP" "Plane" "PowerCons" "ProximalPhalanxOutlineCorrect" "ProximalPhalanxOutlineAgeGroup" "ProximalPhalanxTW" "RefrigerationDevices" "Rock" "ScreenType" "SemgHandGenderCh2" "SemgHandMovementCh2" "SemgHandSubjectCh2" "ShapeletSim" "ShapesAll" "SmallKitchenAppliances" "SmoothSubspace" "SonyAIBORobotSurface1" "SonyAIBORobotSurface2" "StarLightCurves" "Strawberry" "SwedishLeaf" "Symbols" "SyntheticControl" "ToeSegmentation1" "ToeSegmentation2" "Trace" "TwoLeadECG" "TwoPatterns" "UMD" "UWaveGestureLibraryAll" "UWaveGestureLibraryX" "UWaveGestureLibraryY" "UWaveGestureLibraryZ" "Wafer" "Wine" "WordSynonyms" "Worms" "WormsTwoClass" "Yoga" "AconityMINIPrinterLarge" "AconityMINIPrinterSmall" "AllGestureWiimoteX" "AllGestureWiimoteY" "AllGestureWiimoteZ" "AsphaltObstacles" "Colposcopy" "Covid3Month" "DodgerLoopDay" "DodgerLoopGame" "DodgerLoopWeekend" "ElectricDeviceDetection" "FloodModeling1" "FloodModeling2" "FloodModeling3" "GestureMidAirD1" "GestureMidAirD2" "GestureMidAirD3" "GesturePebbleZ1" "GesturePebbleZ2" "MelbournePedestrian" "PickupGestureWiimoteZ" "ShakeGestureWiimoteZ" "SharePriceIncrease")

to_skip_for_now=("PLAID" "Tools" "AsphaltPavementType" "AsphaltRegularity" "KeplerLightCurves" "PhoneHeartbeatSound" "HandOutlines" "Phoneme" "ACSF1" "DiatomSizeReduction" "FiftyWords")

for dataset_name in "${dataset_names[@]}"; do

    for estimator in "${estimators[@]}"; do

        running_experiment_name=$task"_"$dataset_name"_"$estimator
        args=$running_experiment_name
        args=$args$prefix_task$task
        args=$args$prefix_dataset_name$dataset_name
        args=$args$prefix_feature_extractor$estimator

        chmod +x run_exp.sh
        bash run_exp.sh $args

    done

done
