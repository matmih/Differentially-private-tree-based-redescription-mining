/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package dptressexpm.twotreesEM;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

/**
 *
 * @author Matej
 */
public class GenerateSettings {
        public static void main(String args[]){
            ArrayList<String> lines = new ArrayList<>();
            //lines.add("inputFile1 = /home/mmihelci/dprmexp/data/Patient_Claims_aggregated_Sample_2-1_arff.arff");
           // lines.add("inputFile2 = /home/mmihelci/dprmexp/data/Patient_Perso_aggregated_Sample_2-1_arffsparse.arff");
            
           // lines.add("inputFile1 = /home/mmihelci/dprmexp/data/personality_questions_no_id_elapsed_no_NA_OKFilteredVars.arff");
            //lines.add("inputFile2 = /home/mmihelci/dprmexp/data/background_questions_no_scree_major_no_NA_OKFilteredVars.arff");
            
            //lines.add("inputFile1 = /home/mmihelci/dprmexp/data/mammals.arff");
            //lines.add("inputFile2 = /home/mmihelci/dprmexp/data/worldclim.arff");
            
            //lines.add("inputFile1 = /home/mmihelci/dprmexp/data/diagnosesTernary_filtered.arff");
            //lines.add("inputFile2 = /home/mmihelci/dprmexp/data/lab_eventsTernary_filtered.arff");
            
            //lines.add("inputFile1 = /home/mmihelci/dprmexp/data/Patient_Claims_aggregated_Sample_2-4_arff.arff");
            //lines.add("inputFile2 = /home/mmihelci/dprmexp/data/Patient_Perso_aggregated_Sample_2-4_arffsparse.arff");
            
           // lines.add("inputFile1 = /home/mmihelci/dprmexp/data/Patient_Claims_aggregated_Sample_2-8_arff.arff");
           // lines.add("inputFile2 = /home/mmihelci/dprmexp/data/Patient_Perso_aggregated_Sample_2-8_arffsparse.arff");
            
            lines.add("inputFile1 = /home/mmihelci/dprmexp/data/Patient_Claims_aggregated_Sample_2-16_arff.arff");
            lines.add("inputFile2 = /home/mmihelci/dprmexp/data/Patient_Perso_aggregated_Sample_2-16_arffsparse.arff");
            
          /*  inputFile1 = /home/mmihelci/dprmexp/data/diagnosesTernary_filtered.arff
           inputFile2 = /home/mmihelci/dprmexp/data/lab_eventsTernary_filtered.arff
           outputFile = /home/mmihelci/dprmexp/budgetTests/redescriptionsMIMICTernarySupp100MDBB01R6.rr*/

        try{    
           //String output = "F:\\Matej Dokumenti\\Redescription mining with CLUS\\OtherDPRMAlgos\\SettingsFilesRev2\\SettingsCMS2_1Rev2_";
           // String output = "F:\\Matej Dokumenti\\Redescription mining with CLUS\\OtherDPRMAlgos\\SettingsFilesRev2\\SettingsNPASRev2_";
           // String output = "F:\\Matej Dokumenti\\Redescription mining with CLUS\\OtherDPRMAlgos\\SettingsFilesRev2\\SettingsMamamlsRev2_";
           //String output = "F:\\Matej Dokumenti\\Redescription mining with CLUS\\OtherDPRMAlgos\\SettingsFilesRev2\\SettingsMIMICRev2_";
            //String output = "F:\\Matej Dokumenti\\Redescription mining with CLUS\\OtherDPRMAlgos\\SettingsFilesRev2\\SettingsCMS2_4Rev2_";
            //String output = "F:\\Matej Dokumenti\\Redescription mining with CLUS\\OtherDPRMAlgos\\SettingsFilesRev2\\SettingsCMS2_8Rev2_";
            String output = "F:\\Matej Dokumenti\\Redescription mining with CLUS\\OtherDPRMAlgos\\SettingsFilesRev2\\SettingsCMS2_16Rev2_";
            
           //String outputAlgo = "outputFile = /home/mmihelci/dprmexp/budgetTests/redescriptionsCMS2_1Rev2";
           // String outputAlgo = "outputFile = /home/mmihelci/dprmexp/budgetTests/redescriptionsNPASRev2";
            //String outputAlgo = "outputFile = /home/mmihelci/dprmexp/budgetTests/redescriptionsMammalsRev2";
            //String outputAlgo = "outputFile = /home/mmihelci/dprmexp/budgetTests/redescriptionsMIMICRev2";
           // String outputAlgo = "outputFile = /home/mmihelci/dprmexp/budgetTests/redescriptionsCMS2_4Rev2";
            //String outputAlgo = "outputFile = /home/mmihelci/dprmexp/budgetTests/redescriptionsCMS2_8Rev2";
            String outputAlgo = "outputFile = /home/mmihelci/dprmexp/budgetTests/redescriptionsCMS2_16Rev2";
            int algoType = 2;  
            int stabilized = 1;
            outputAlgo+="A"+algoType;
            lines.add(outputAlgo);
            lines.add("weightSplit = 0.5");
            lines.add("weightNoise = 0.5");
            lines.add("numInitTrials = 1");
            lines.add("numIterations = 20");
            lines.add("maxTreeDepth = 4");
            lines.add("maxMCMCIteration = 10000");
            lines.add("equilibriumTreshold = 0.005");
            lines.add("maxRedSupport = 0.8");
            lines.add("minRedSupport = 100");
            lines.add("minJaccard = 0.1");
            lines.add("maxPval = 0.01");
            lines.add("maxNumClauses = 3");
            lines.add("totalPrivacyBudget = 0.01");
            FileWriter fw = null;
            
            //algoType - 0 expMech, 1 - MCMCTrees, 2 - Madre, 3 - MadreB
            
            for(int i=0;i<100;i++){
                //output = "F:\\Matej Dokumenti\\Redescription mining with CLUS\\OtherDPRMAlgos\\SettingsFilesRev2\\SettingsCMS2_1Rev2_";
                //output = "F:\\Matej Dokumenti\\Redescription mining with CLUS\\OtherDPRMAlgos\\SettingsFilesRev2\\SettingsNPASRev2_";
                //output = "F:\\Matej Dokumenti\\Redescription mining with CLUS\\OtherDPRMAlgos\\SettingsFilesRev2\\SettingsMammalsRev2_";
               // output = "F:\\Matej Dokumenti\\Redescription mining with CLUS\\OtherDPRMAlgos\\SettingsFilesRev2\\SettingsMIMICRev2_";
                //output = "F:\\Matej Dokumenti\\Redescription mining with CLUS\\OtherDPRMAlgos\\SettingsFilesRev2\\SettingsCMS2_4Rev2_";
                //output = "F:\\Matej Dokumenti\\Redescription mining with CLUS\\OtherDPRMAlgos\\SettingsFilesRev2\\SettingsCMS2_8Rev2_";
                output = "F:\\Matej Dokumenti\\Redescription mining with CLUS\\OtherDPRMAlgos\\SettingsFilesRev2\\SettingsCMS2_16Rev2_";
                //outputAlgo = "outputFile = /home/mmihelci/dprmexp/budgetTests/ResultsDAMIRev2/redescriptionsCMS2_1Rev2";
                //outputAlgo = "outputFile = /home/mmihelci/dprmexp/budgetTests/ResultsDAMIRev2/redescriptionsNPASRev2";
               // outputAlgo = "outputFile = /home/mmihelci/dprmexp/budgetTests/ResultsDAMIRev2/redescriptionsMammalsRev2";
                //outputAlgo = "outputFile = /home/mmihelci/dprmexp/budgetTests/ResultsDAMIRev2/redescriptionsMIMICRev2";
                //outputAlgo = "outputFile = /home/mmihelci/dprmexp/budgetTests/ResultsDAMIRev2/redescriptionsCMS2_4Rev2";
                //outputAlgo = "outputFile = /home/mmihelci/dprmexp/budgetTests/ResultsDAMIRev2/redescriptionsCMS2_8Rev2";
                outputAlgo = "outputFile = /home/mmihelci/dprmexp/budgetTests/ResultsDAMIRev2/redescriptionsCMS2_16Rev2";
                //outputAlgo+="R"+(i+1)+"A"+algoType+"S"+stabilized+".rr";
                outputAlgo+="R"+(i+1)+"A"+algoType+"B"+stabilized+".rr";
                lines.set(2, outputAlgo);
                //output+=(i+1)+"A"+algoType+"S"+stabilized+".set";
                output+=(i+1)+"A"+algoType+"B"+stabilized+".set";
                   fw = new FileWriter(output); 
                  for(int j=0;j<lines.size();j++)
                      fw.write(lines.get(j)+"\n");
                  
                  fw.close();
            }
             
        }
        catch(IOException e){
           e.printStackTrace();
        }
            
















            
        }
}
