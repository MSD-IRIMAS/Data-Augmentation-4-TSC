#!/bin/bash

task="generate_data"

prefix_task=" task="
prefix_dataset_name=" dataset_name="
prefix_augmentation_method=" generate_data.method="
prefix_distance=" generate_data.distance="

dataset_names=("Adiac" "ArrowHead" "Beef" "BeetleFly" "BirdChicken" "BME" "Car" "CBF" "Chinatown" "ChlorineConcentration" "CinCECGTorso" "Coffee" "Computers" "CricketX" "CricketY" "CricketZ" "Crop" "DistalPhalanxOutlineCorrect" "DistalPhalanxOutlineAgeGroup" "DistalPhalanxTW" "Earthquakes" "ECG200" "ECG5000" "ECGFiveDays" "ElectricDevices" "EOGHorizontalSignal" "EOGVerticalSignal" "EthanolLevel" "FaceAll" "FaceFour" "FacesUCR" "Fish" "FordA" "FordB" "FreezerRegularTrain" "FreezerSmallTrain" "GunPoint" "GunPointAgeSpan" "GunPointMaleVersusFemale" "GunPointOldVersusYoung" "Ham" "Haptics" "Herring" "HouseTwenty" "InlineSkate" "InsectEPGRegularTrain" "InsectEPGSmallTrain" "InsectWingbeatSound" "ItalyPowerDemand" "LargeKitchenAppliances" "Lightning2" "Lightning7" "Mallat" "Meat" "MedicalImages" "MiddlePhalanxOutlineCorrect" "MiddlePhalanxOutlineAgeGroup" "MiddlePhalanxTW" "MixedShapesRegularTrain" "MixedShapesSmallTrain" "MoteStrain" "NonInvasiveFetalECGThorax1" "NonInvasiveFetalECGThorax2" "OliveOil" "OSULeaf" "PhalangesOutlinesCorrect" "PigAirwayPressure" "PigArtPressure" "PigCVP" "Plane" "PowerCons" "ProximalPhalanxOutlineCorrect" "ProximalPhalanxOutlineAgeGroup" "ProximalPhalanxTW" "RefrigerationDevices" "Rock" "ScreenType" "SemgHandGenderCh2" "SemgHandMovementCh2" "SemgHandSubjectCh2" "ShapeletSim" "ShapesAll" "SmallKitchenAppliances" "SmoothSubspace" "SonyAIBORobotSurface1" "SonyAIBORobotSurface2" "StarLightCurves" "Strawberry" "SwedishLeaf" "Symbols" "SyntheticControl" "ToeSegmentation1" "ToeSegmentation2" "Trace" "TwoLeadECG" "TwoPatterns" "UMD" "UWaveGestureLibraryAll" "UWaveGestureLibraryX" "UWaveGestureLibraryY" "UWaveGestureLibraryZ" "Wafer" "Wine" "WordSynonyms" "Worms" "WormsTwoClass" "Yoga" "AconityMINIPrinterLarge" "AconityMINIPrinterSmall" "AllGestureWiimoteX" "AllGestureWiimoteY" "AllGestureWiimoteZ" "AsphaltObstacles" "Colposcopy" "Covid3Month" "DodgerLoopDay" "DodgerLoopGame" "DodgerLoopWeekend" "ElectricDeviceDetection" "FloodModeling1" "FloodModeling2" "FloodModeling3" "GestureMidAirD1" "GestureMidAirD2" "GestureMidAirD3" "GesturePebbleZ1" "GesturePebbleZ2" "MelbournePedestrian" "PickupGestureWiimoteZ" "ShakeGestureWiimoteZ" "SharePriceIncrease")

to_skip_for_now=("PLAID" "Tools" "AsphaltPavementType" "AsphaltRegularity" "KeplerLightCurves" "PhoneHeartbeatSound" "DiatomSizeReduction" "FiftyWords" "HandOutlines" "Phoneme" "ACSF1")

json_file="methods.json"
methods=$(jq -r 'keys[]' "$json_file")

for dataset_name in "${dataset_names[@]}"; do

    for method in $methods; do

        values=$(jq -c --arg method "$method" '.[$method]' "$json_file")

        if [ "$values" == "[]" ]; then
            args=$prefix_task$task
            args=$args$prefix_dataset_name$dataset_name
            args=$args$prefix_augmentation_method$method

            chmod +x run_exp.sh
            bash run_exp.sh $args
        else
            echo "Method: $method (With parameters)"
            jq -c --arg method "$method" '.[$method][]' "$json_file" | while read -r params; do
                distance=$(echo "$params" | jq -r '.distance')

                if echo "$params" | jq -e 'has("window")' > /dev/null; then
                    extra_param=$(echo "$params" | jq -r '.window')
                    extra_key="window"
                elif echo "$params" | jq -e 'has("reach")' > /dev/null; then
                    extra_param=$(echo "$params" | jq -r '.reach')
                    extra_key="reach"
                elif echo "$params" | jq -e 'has("c")' > /dev/null; then
                    extra_param=$(echo "$params" | jq -r '.c')
                    extra_key="c"
                else
                    extra_param=""
                    extra_key=""
                fi

                if [ -z "$extra_param" ]; then
                    args=$prefix_task$task
                    args=$args$prefix_dataset_name$dataset_name
                    args=$args$prefix_augmentation_method$method
                    args=$args$prefix_distance$distance

                    chmod +x run_exp.sh
                    bash run_exp.sh $args
                else
                    space=" "
                    equal="="
                    prefix_extra_key="generate_data."

                    args=$prefix_task$task
                    args=$args$prefix_dataset_name$dataset_name
                    args=$args$prefix_augmentation_method$method
                    args=$args$prefix_distance$distance
                    args=$args$space$prefix_extra_key$extra_key$equal$extra_param

                    chmod +x run_exp.sh
                    bash run_exp.sh $args
                fi
            done
        fi
    done
done
