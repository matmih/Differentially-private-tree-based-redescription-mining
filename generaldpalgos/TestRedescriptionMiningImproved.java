/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package generaldpalgos;

import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Random;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

/**
 *
 * @author Matej
 */
public class TestRedescriptionMiningImproved {
    
    public static void main(String args[]){
          
         if(args.length==0){
            System.err.println("A path to the settings file must be provided!");
            System.err.println("Terminating execution!");
            System.exit(-1);
        }
       
         int algo = 0;
         
         if(args.length==2){
             algo = Integer.parseInt(args[1]);
         }
         
         long startTime = 0, endTime = 0, startTimeTotal = 0, endTimeTotal = 0;
         
   
        File settingsPath = new File(args[0].trim());
        
        Settings set = new Settings(settingsPath);
        
        String inputFile1 = set.inputFile1;
        String inputFile2 = set.inputFile2;
        Instances dataW1 = null, dataW2 = null, currentData = null;  
        DecTreeExpMech dt1=null, dt2=null;
        DecTreeWholeDp dtt1=null, dtt2=null;
        
        try{
                dataW1 = (new ConverterUtils.DataSource(inputFile1)).getDataSet();
                dataW2 = (new ConverterUtils.DataSource(inputFile2)).getDataSet();
                
                   startTimeTotal =   System.currentTimeMillis();
                
                ArrayList<Redescription> AllRedescriptions = new ArrayList<>();
                
                int numIterations = set.numIterations;//80;
                
                 int numAttrs = dataW1.numAttributes();
                 int numAttrs1 = dataW2.numAttributes();
                 
                 numIterations = Math.min(numIterations, numAttrs+numAttrs1);
                
                
                double epsilon = set.budget/(((double)numIterations)*2+1);
                System.out.println("Epsilon: "+epsilon);

                HashSet<Integer> usedIndex = new HashSet<>();
                
                
                //create initial targets
               
                Random rand = new Random();
               
               int minSupport = 0, maxSupport = 0;
               
               if(set.minRedSupport<=1.0)
                   minSupport = (int)(dataW1.numInstances()*set.minRedSupport);
               else minSupport = (int)set.minRedSupport;
               
               if(set.maxRedSupport<=1.0)
                   maxSupport = (int)(dataW1.numInstances()*set.maxRedSupport);
               else maxSupport = (int)set.maxRedSupport;
               
               if(minSupport>maxSupport){
                   System.err.println("minSupport>maxSupport, change settings file!");
                   System.exit(-2);
               }
                   
                int totalAttrs = numAttrs+numAttrs1;
               double prob = ((double)numAttrs)/((double) totalAttrs);//probability of choosing attr from view1
               Instances data1 = null;
               int side = 0;
			   
               if(algo == 0)
                 dt1 = new DecTreeExpMech();
               else dtt1 = new DecTreeWholeDp();
               
                double probSim = rand.nextDouble();
                 if(probSim<=prob){
                     System.out.println("Choise from view1!");
                 }
                 else System.out.println("Choise from view2!");
               
             for(int it=0;it<=numIterations;it++){
                 System.out.println("Iteration: "+it+"/ "+numIterations);
                 
                 //generate a random number deciding view from which to take an attribute

               int rind = 0;
               if(it == 0){
                        if(probSim<=prob)
                            rind = rand.nextInt(dataW1.numAttributes());
                        else  rind = rand.nextInt(dataW2.numAttributes());
               
                Attribute r = null;
                if(probSim<=prob)
                r = dataW1.attribute(rind);
                else r = dataW2.attribute(rind);
  
                int numValues = r.numValues();//number of classess
                System.out.println("Num classes: "+numValues);
                
                if(probSim<=prob)
                data1 = new Instances(dataW1);
                else data1 = new Instances(dataW2);
               

                ArrayList<String> values = new ArrayList<>(); /* FastVector is now deprecated. Users can use any java.util.List */
       
                startTime =  System.currentTimeMillis();
                 
                if(!r.isNumeric()){ 
                  for(int i=0;i<numValues;i++)
                        values.add("c"+i);
                  data1.insertAttributeAt(new Attribute("NewNominal", values), data1.numAttributes());
                  data1.setClassIndex(data1.numAttributes()-1);
                  
                   for(int i=0;i<data1.numInstances();i++){
                    //  data1.get(i).
                    int in = 0;
                    String val = data1.get(i).stringValue(r.index());
                    for(int zz = 0; zz<r.numValues();zz++)
                        if(val.trim().equals(r.value(zz).trim())){
                            in = zz;
                            break;
                        }
                      data1.get(i).setClassValue(values.get(in));
                  }
                  
                }
                else{
                    
                    HashSet<Double> attrV = new HashSet<>();
                    ArrayList<Double> attrValues = null;
                    
                    double perc = 0.92;
                    int numtrials = 0;
                    
                    for(int s = 0; s<perc*data1.numInstances();s++){
                        int ind = rand.nextInt(data1.numInstances());
                        numtrials++;
                        if(perc>0.9)
                            ind = s;
                        if(!data1.get(ind).isMissing(r))
                             attrV.add(data1.get(ind).value(r));
                        else if(perc<0.9) s--;
                        if(numtrials >= data1.numInstances())
                            break;
                    }
                      
                      attrValues = new ArrayList(attrV);
                      Collections.sort(attrValues);
                      double vp[] = new double[attrValues.size()];
                      
                      System.out.println("vp: ");
                      for(int s=0;s<attrValues.size();s++){
                          System.out.print(attrValues.get(s)+" ");
                          vp[s] = attrValues.get(s);
                      }
                      System.out.println();
                      
                      double n = Math.pow(data1.numInstances(), -0.33);
                      DescriptiveStatistics da = new DescriptiveStatistics(vp);
                      double iqr = da.getPercentile(75) - da.getPercentile(25);
                      double h = 2*iqr*n;
                      
                      System.out.println("h: "+h);
                      
                      if(h == 0 || h == Double.NaN || h == Double.POSITIVE_INFINITY)
                          continue;
                      
                      ArrayList<Double> splittingPoints = new ArrayList<>();
                      
                      double tmpInit = attrValues.get(0);
                      for(int s=1;s<attrValues.size();s++){
                              if(Math.abs(attrValues.get(s)-tmpInit)>h){
                                  splittingPoints.add(attrValues.get(s));
                                  tmpInit = attrValues.get(s);
                              }
                      }
                      
                     System.out.println("attrValues size: "+attrValues.size());
                     System.out.println("n: "+n);
                      
                       for(int s=0;s<splittingPoints.size()+2;s++){
                        values.add("c"+s);
                       }
                       System.out.println();
                  data1.insertAttributeAt(new Attribute("NewNominal", values), data1.numAttributes());
                  data1.setClassIndex(data1.numAttributes()-1);
                  
                  System.out.println("Splitting point size: "+splittingPoints.size());
                  int numAssigned = 0;
                  HashMap<Integer,Integer> classCounts = new HashMap<>();
                  
                    for(int s=0;s<data1.numInstances();s++){
                        
                        if(data1.get(s).value(r)<=splittingPoints.get(0)){
                             data1.get(s).setClassValue(values.get(0));
                             numAssigned++;
                             if(!classCounts.containsKey(0))
                                 classCounts.put(0, 1);
                             else classCounts.put(0,classCounts.get(0)+1);
                        }
                        else if(data1.get(s).value(r)>splittingPoints.get(splittingPoints.size()-1)){
                            data1.get(s).setClassValue(values.get(values.size()-1));
                            numAssigned++;
                            
                            if(!classCounts.containsKey(values.size()-1))
                                 classCounts.put(values.size()-1, 1);
                             else classCounts.put(values.size()-1,classCounts.get(values.size()-1)+1);
                            
                        }
                        else{
                            int a = 0;
                            for(int s1 = 1;s1<splittingPoints.size();s1++)
                                if(data1.get(s).value(r)<=splittingPoints.get(s1)){
                                      data1.get(s).setClassValue(values.get(s1));
                                         numAssigned++;
                                         a = 1;
                                         if(!classCounts.containsKey(s1))
                                                  classCounts.put(s1, 1);
                                          else classCounts.put(s1,classCounts.get(s1)+1);
                                         break;
                                }
                            if(a == 0){
                             data1.get(s).setClassValue(values.get(rand.nextInt(values.size())));
                             System.out.println("su: "+s);
                            }
                        }

                    }
                

                     System.out.println("Num assigned: "+numAssigned); 
                     
                     System.out.println("Class assignement: ");
                     for(int c:classCounts.keySet())
                         System.out.println(c+" "+classCounts.get(c));
                    }
                
                 endTime =  System.currentTimeMillis();
                 
                 long elapsedTime = endTime - startTime;
                System.out.println("Target creation time: "+elapsedTime/(1000.0)+"s");
                    
                System.out.println("CA: "+data1.classAttribute().toString()); 
                
                  startTime = System.currentTimeMillis();

              if(algo == 0){
                dt1.setEpsilon(epsilon+"");
                dt1.setMaxDepth(set.maxTreeDepth);
                dt1.buildClassifier(data1);
              }
              else{
                dtt1.setEpsilon(epsilon+"");
                dtt1.setMaxDepth(set.maxTreeDepth);
                dtt1.setMaxIteration(set.maxMCMCIteration);
                dtt1.setEquilibriumThreshold(set.equilibriumTreshold);
                dtt1.buildClassifier(data1);  
              }
                 
                    endTime = System.currentTimeMillis();
                 
                   elapsedTime = endTime - startTime;
                   
                    System.out.println("Tree creation time: "+elapsedTime/(1000.0)+"s");
                    side = 1;
               }//first iteration, create random targets
               else{ //uses previous tree as target
                  if(side == 0){
                  if(algo == 0)
                   dt2.renumerateLeaves(dt2.getRoot());
                  else dtt2.renumerateLeaves(dtt2.getRoot());
                  
                   Instances data2 = null;
                   if(probSim<=prob)
                           data2 = new Instances(dataW1);
                   else data2 = new Instances(dataW2);
                    data2.insertAttributeAt(new Attribute("NewNominal", new ArrayList<>()), data2.numAttributes());
                    data2.setClassIndex(data2.numAttributes()-1);
                   if(algo == 0){
                         data1 = dt2.addTargets(data1);
                         data2 = dt2.copyTargets(data1,data2);
                   }
                   else{
                       data1 = dtt2.addTargets(data1);
                       data2 = dtt2.copyTargets(data1, data2);
                   }
                   data1 = data2;
                   
                  if(algo == 0){ 
                   dt1 = new DecTreeExpMech();
                   dt1.setEpsilon(epsilon+"");
                   dt1.setMaxDepth(set.maxTreeDepth);
                   dt1.buildClassifier(data1);
                  }
                 else{
                dtt1 = new DecTreeWholeDp();
                dtt1.setEpsilon(epsilon+"");
                dtt1.setMaxDepth(set.maxTreeDepth);
                dtt1.setMaxIteration(set.maxMCMCIteration);
                dtt1.setEquilibriumThreshold(set.equilibriumTreshold);
                dtt1.buildClassifier(data1);  
              }
                  
                 TwoTrees tt = new TwoTrees();
                tt.view1Data = dataW1;
                tt.view2Data = dataW2;
                tt.setMinSupportReal(minSupport);
                tt.setMaxDepth(set.maxTreeDepth);
                tt.setminNodeSize(minSupport);
                tt.setEpsilon(epsilon+"");
                tt.setWeights(set.w1, set.w2);
                tt.setMaxIteration(set.maxMCMCIteration);
                tt.setEquilibriumThreshold(set.equilibriumTreshold);
                tt.setMaxIterationRed(500);
                tt.setNumClausesRed(3);
                tt.setMaxSupport(maxSupport);
                tt.setMinSupport(minSupport);
                tt.setMinJaccard(set.minJaccard);
                tt.setMaxPval(set.maxPval);
                if(algo == 0){
                    if(probSim<=prob){
                         tt.setRootFirst(dt1.getRoot());
                         tt.setRootSecond(dt2.getRoot());
                    }
                    else{
                         tt.setRootFirst(dt2.getRoot());
                         tt.setRootSecond(dt1.getRoot()); 
                    }
                }
                else{
                    if(probSim<=prob){
                    tt.setRootFirst(dtt1.getRoot());
                    tt.setRootSecond(dtt2.getRoot());
                    }
                    else{
                       tt.setRootFirst(dtt2.getRoot());
                       tt.setRootSecond(dtt1.getRoot());  
                    }
                }
                ArrayList<Redescription> tmp = null;
                
                startTime = System.currentTimeMillis();
                

                tmp = tt.createRedescriptionsNoisy(dataW1, dataW2, set.numClauses, 0);
               
                AllRedescriptions.addAll(tmp);
                
                 endTime = System.currentTimeMillis();
             
                 double elapsedTime = endTime - startTime;
                 System.out.println("Redescription creation time: "+elapsedTime/(1000.0)+"s");
                 side = 1-side;
                  }
                  else{
                      
                       side = 1-side;
                       if(algo == 0)
                        dt1.renumerateLeaves(dt1.getRoot());
                       else dtt1.renumerateLeaves(dtt1.getRoot());
                if(algo == 0)
                  data1 = dt1.addTargets(data1);
                else data1 = dtt1.addTargets(data1);
                Instances data2 = null;
                if(probSim<=prob)
                        data2 = new Instances(dataW2);
                else data2 = new Instances(dataW1);
                if(algo == 0)
                     data2 = dt1.copyTargets(data1,data2);
                else data2 = dtt1.copyTargets(data1, data2);
                data1 = data2;
               
                if(algo == 0){
                dt2 = new DecTreeExpMech();
                dt2.setEpsilon(epsilon+"");
                dt2.setMaxDepth(set.maxTreeDepth);
                dt2.buildClassifier(data1);
                System.out.println("Classifier2: ");
                System.out.println(dt2);
                }
                else{
                dtt2 = new DecTreeWholeDp();
                dtt2.setEpsilon(epsilon+"");
                dtt2.setMaxDepth(set.maxTreeDepth);
                dtt2.setMaxIteration(set.maxMCMCIteration);
                dtt2.setEquilibriumThreshold(set.equilibriumTreshold);
                dtt2.buildClassifier(data1);  
              }
                
                TwoTrees tt = new TwoTrees();
                tt.view1Data = dataW1;
                tt.view2Data = dataW2;
                
                tt.setMinSupportReal(minSupport);
                tt.setMaxDepth(set.maxTreeDepth);
                tt.setminNodeSize(minSupport);
                tt.setEpsilon(epsilon+"");
                tt.setWeights(set.w1, set.w2);
                tt.setMaxIteration(set.maxMCMCIteration);
                tt.setEquilibriumThreshold(set.equilibriumTreshold);
                tt.setMaxIterationRed(500);
                tt.setNumClausesRed(3);
                tt.setMaxSupport(maxSupport);
                tt.setMinSupport(minSupport);
                tt.setMinJaccard(set.minJaccard);
                tt.setMaxPval(set.maxPval);
                if(algo == 0){
                   if(probSim<=prob){
                      tt.setRootFirst(dt1.getRoot());
                      tt.setRootSecond(dt2.getRoot());
                   }
                   else{
                        tt.setRootFirst(dt2.getRoot());
                        tt.setRootSecond(dt1.getRoot());
                   }
                }
                else{
                    if(probSim<=prob){
                     tt.setRootFirst(dtt1.getRoot());
                    tt.setRootSecond(dtt2.getRoot());
                    }
                    else{
                         tt.setRootFirst(dtt2.getRoot());
                        tt.setRootSecond(dtt1.getRoot());
                    }
                }
                System.out.println();
                System.out.println("Redescription creation started...");
                               
                System.out.println();
                System.out.println();
                System.out.println("Creating redescriptions directly using nosy counts: ");
                ArrayList<Redescription> tmp = null;
                
                startTime = System.currentTimeMillis();
                

                tmp = tt.createRedescriptionsNoisy(dataW1, dataW2, set.numClauses, 0);
               
                AllRedescriptions.addAll(tmp);
                
                 endTime = System.currentTimeMillis();
             
                 double elapsedTime = endTime - startTime;
                   
              System.out.println("Redescription creation time: "+elapsedTime/(1000.0)+"s");
                  }
               }
             }
             
             
             Collections.sort(AllRedescriptions, Collections.reverseOrder());
             

             System.out.println();
             System.out.println();
             System.out.println("Totali produced: "+AllRedescriptions.size()+" redescriptions");
             System.out.println("All produced redescriptions: ");
             
             FileWriter fw = new FileWriter(set.outputFile);
             String rereMiOut = set.outputFile.split(".rr")[0].trim()+".queries";
             FileWriter fw1 = new FileWriter(rereMiOut);
             int rCount = 0;
             
             for(int i=0;i<AllRedescriptions.size();i++){
                 System.out.println(AllRedescriptions.get(i).queryStrings.get(0));
                 System.out.println(AllRedescriptions.get(i).queryStrings.get(1));
                 System.out.println("JS: "+AllRedescriptions.get(i).Jaccard);
                 System.out.println("p-value: "+AllRedescriptions.get(i).pval);
                 System.out.println("query support sizes: "+AllRedescriptions.get(i).querySupports.get(0)+" "+AllRedescriptions.get(i).querySupports.get(1));
                 System.out.println("support size: "+AllRedescriptions.get(i).support);
                 System.out.println("query union size: "+AllRedescriptions.get(i).supportUnion);
                 System.out.println("\n\n");
                 
                  if(i == 0){
                     fw.write("Redescriptions: \n\n");
                     fw1.write("rid	query_LHS	query_RHS	acc	pval	card_Exo	card_Eox	card_Exx	card_Eoo	\n");
                  }
                     
                 fw.write("q1: "+AllRedescriptions.get(i).queryStrings.get(0)+"\n");
                 fw.write("q2: "+AllRedescriptions.get(i).queryStrings.get(1)+"\n");
                 fw.write("J: "+AllRedescriptions.get(i).Jaccard+"\n");
                 fw.write("p: "+AllRedescriptions.get(i).pval+"\n");
                 fw.write("Redescription support size: "+AllRedescriptions.get(i).support+"\n");
                 fw.write("Query support union size: "+AllRedescriptions.get(i).supportUnion+"\n");
                 fw.write("\n\n");
               
                 String rq1 = AllRedescriptions.get(i).queryStringsReReMi.get(0);
                 String rq2 = AllRedescriptions.get(i).queryStringsReReMi.get(1);
                 
                 fw1.write("r"+(++rCount)+"\t"+rq1+"\t"+rq2+"\t"+AllRedescriptions.get(i).Jaccard+"\t"+AllRedescriptions.get(i).pval+"\t"+AllRedescriptions.get(i).querySupports.get(0)+"\t"+AllRedescriptions.get(i).querySupports.get(1)+"\t"+AllRedescriptions.get(i).support+"\t"+(dataW1.numInstances()-AllRedescriptions.get(i).supportUnion)+"\n");                
             }
             
             endTimeTotal =   System.currentTimeMillis();
              long elapsedTime = endTimeTotal - startTimeTotal;
              fw.write("\nTime: "+(elapsedTime/(1000.0))+"s");
             
             fw.close();
             fw1.close();
             
        }
        catch(Exception e){
            e.printStackTrace();
        }
    }
    
}
