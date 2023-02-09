/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package dptressexpm.twotreesEM;

import java.io.FileWriter;
import java.io.IOException;

/**
 *
 * @author Matej
 */
public class CreateParametersFille {
    public static void main(String args[]){
        
        String imena[] = {"ExpMechNS","ExpMechS","MCMCTreeNS","MCMCTreeS","Madre","MadreB"};
        
        try{
            FileWriter fw = null;
            for(int i=0;i<6;i++){
                fw = new FileWriter("F:\\Matej Dokumenti\\Redescription mining with CLUS\\OtherDPRMAlgos\\SettingsFilesRev2\\"+"parametersCMS2_16"+(imena[i])+".txt");
                
                for(int j=0;j<100;j++){
                    if(i==0){
                        //String set = "SettingsCMS2_1Rev2_"+(j+1)+"A0S0.set";
                        //String set = "SettingsNPASRev2_"+(j+1)+"A0S0.set";
                       // String set = "SettingsMammalsRev2_"+(j+1)+"A0S0.set";
                       // String set = "SettingsMIMICRev2_"+(j+1)+"A0S0.set";
                         //String set = "SettingsCMS2_4Rev2_"+(j+1)+"A0S0.set";
                         //String set = "SettingsCMS2_8Rev2_"+(j+1)+"A0S0.set";
                         String set = "SettingsCMS2_16Rev2_"+(j+1)+"A0S0.set";
                        fw.write(set);
                        fw.write(" "+0+" "+(j+1)+"\n");
                    }
                    else if(i==1){
                        //String set = "SettingsCMS2_1Rev2_"+(j+1)+"A0S1.set";
                        //String set = "SettingsNPASRev2_"+(j+1)+"A0S1.set";
                        //String set = "SettingsMammalsRev2_"+(j+1)+"A0S1.set";
                        //String set = "SettingsMIMICRev2_"+(j+1)+"A0S1.set";
                        //String set = "SettingsCMS2_4Rev2_"+(j+1)+"A0S1.set";
                        //String set = "SettingsCMS2_8Rev2_"+(j+1)+"A0S1.set";
                        String set = "SettingsCMS2_16Rev2_"+(j+1)+"A0S1.set";
                        fw.write(set);
                        fw.write(" "+0+" "+(j+1)+"\n");
                    }
                    else if(i==2){
                        //String set = "SettingsCMS2_1Rev2_"+(j+1)+"A1S0.set";
                       // String set = "SettingsNPASRev2_"+(j+1)+"A1S0.set";
                        //String set = "SettingsMammalsRev2_"+(j+1)+"A1S0.set";
                       // String set = "SettingsMIMICRev2_"+(j+1)+"A1S0.set";
                        //String set = "SettingsCMS2_4Rev2_"+(j+1)+"A1S0.set";
                        //String set = "SettingsCMS2_8Rev2_"+(j+1)+"A1S0.set";
                        String set = "SettingsCMS2_16Rev2_"+(j+1)+"A1S0.set";
                        fw.write(set);
                        fw.write(" "+1+" "+(j+1)+"\n");
                    }
                    else if(i==3){
                        // String set = "SettingsCMS2_1Rev2_"+(j+1)+"A1S1.set";
                        // String set = "SettingsNPASRev2_"+(j+1)+"A1S1.set";
                        // String set = "SettingsMammalsRev2_"+(j+1)+"A1S1.set";
                         //String set = "SettingsMIMICRev2_"+(j+1)+"A1S1.set";
                         //String set = "SettingsCMS2_4Rev2_"+(j+1)+"A1S1.set";
                        // String set = "SettingsCMS2_8Rev2_"+(j+1)+"A1S1.set";
                         String set = "SettingsCMS2_16Rev2_"+(j+1)+"A1S1.set";
                        fw.write(set);
                        fw.write(" "+1+" "+(j+1)+"\n");
                    }
                     else if(i==4){
                        // String set = "SettingsCMS2_1Rev2_"+(j+1)+"A2B0.set";
                         //String set = "SettingsNPASRev2_"+(j+1)+"A2B0.set";
                         //String set = "SettingsMammalsRev2_"+(j+1)+"A2B0.set";
                         //String set = "SettingsMIMICRev2_"+(j+1)+"A2B0.set";
                        // String set = "SettingsCMS2_4Rev2_"+(j+1)+"A2B0.set";
                        // String set = "SettingsCMS2_8Rev2_"+(j+1)+"A2B0.set";
                         String set = "SettingsCMS2_16Rev2_"+(j+1)+"A2B0.set";
                        fw.write(set);
                        fw.write(" "+0+" "+(j+1)+"\n");
                    }
                     else if(i==5){
                         //String set = "SettingsCMS2_1Rev2_"+(j+1)+"A2B1.set";
                         //String set = "SettingsNPASRev2_"+(j+1)+"A2B1.set";
                         //String set = "SettingsMammalsRev2_"+(j+1)+"A2B1.set";
                         //String set = "SettingsMIMICRev2_"+(j+1)+"A2B1.set";
                         //String set = "SettingsCMS2_4Rev2_"+(j+1)+"A2B1.set";
                         //String set = "SettingsCMS2_8Rev2_"+(j+1)+"A2B1.set";
                         String set = "SettingsCMS2_16Rev2_"+(j+1)+"A2B1.set";
                        fw.write(set);
                        fw.write(" "+0+" "+(j+1)+"\n");
                    }
                }
                
                fw.close();
            }
        }
        catch(IOException e){
            e.printStackTrace();
        }
            
    }
}
