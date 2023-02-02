/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package generaldpalgos;


import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import org.apache.commons.math3.distribution.BinomialDistribution;
import weka.core.Instances;

/**
 *
 * @author Matej Mihelcic, University of Eastern Finland implemented the dptressxpm.twotreesEM package as an extension of the 
 * original tree package decTreeWholeDP created for the manuscript: "Embedding differential privacy in decision tree
algorithm with different depths" by authors Xuanyu BAI , Jianguo YAO*, Mingxuan YUAN, Ke DENG, Xike Xie and Haibing Guan.
 */
public class Redescription implements Comparator<Redescription>, Comparable<Redescription> {
    
    public ArrayList<String> queryStrings;
    public ArrayList<String> queryStringsReReMi;
    public ArrayList<ArrayList<Node>> redNodes;
    public ArrayList<HashSet<Integer>> queryEntities;
    public double Jaccard, pval, mn;
    public int support, supportUnion;
    public ArrayList<Integer> querySupports;
    
    
    public Redescription(){
        queryStrings = new ArrayList<>(2);
        queryStringsReReMi = new ArrayList<>(2);
        redNodes = new ArrayList<>(2);
         redNodes.add(new ArrayList<>(2)); redNodes.add(new ArrayList<>(2));
        querySupports = new ArrayList<>(2);
        queryEntities = new ArrayList<>(2);
    }
    
    public Redescription(int numClauses){
        queryStrings = new ArrayList(2);
        queryStringsReReMi = new ArrayList(2);
        redNodes = new ArrayList<>(2);
        querySupports = new ArrayList<>(2);
        queryEntities = new ArrayList<>(2);
        
        redNodes.add(new ArrayList<>(numClauses)); redNodes.add(new ArrayList<>(numClauses));
        for(int i=0;i<numClauses;i++){
            redNodes.get(0).add(null);
            redNodes.get(1).add(null);
                    }
    }
    
    public Redescription(Redescription tmp){
        queryStrings = new ArrayList(tmp.queryStrings.size());
        queryStringsReReMi = new ArrayList(tmp.queryStringsReReMi.size());
        redNodes = new ArrayList<>(tmp.redNodes.size());
        querySupports = new ArrayList<>(tmp.querySupports.size());
        queryEntities = new ArrayList<>(tmp.queryEntities.size());
        
        Jaccard = tmp.Jaccard;
        pval = tmp.pval;
        support = tmp.support;
        supportUnion = tmp.supportUnion;
        
        for(int i=0;i<tmp.queryStrings.size();i++)
            queryStrings.add(tmp.queryStrings.get(i));
        
        for(int i=0;i<tmp.queryStringsReReMi.size();i++)
            queryStringsReReMi.add(tmp.queryStringsReReMi.get(i));
        
        for(int i=0;i<tmp.queryEntities.size();i++){
            queryEntities.add(new HashSet<>());
            queryEntities.get(i).addAll(tmp.queryEntities.get(i));
        }
        
        for(int i=0;i<tmp.querySupports.size();i++){
            querySupports.add(tmp.querySupports.get(i));
        }
        
        for(int i =0; i<tmp.redNodes.size();i++){
           redNodes.add(new ArrayList<>());
           redNodes.get(i).addAll(tmp.redNodes.get(i));
        }
        
    }
    
    
    public int compareTo(Redescription r) {
        if(this.Jaccard>r.Jaccard)
            return 1;
        else if(this.Jaccard<r.Jaccard)
            return -1;
        return 0;
   }

   // Overriding the compare method to sort the age 
   public int compare(Redescription r, Redescription r1) {
      if(r.Jaccard>r1.Jaccard)
          return 1;
      else if(r.Jaccard<r1.Jaccard)
          return -1;
      return 0;
   }
   
   public String createElConjunctionStringReReMiCodesHelp(Node n){
      String tmp = "";
      
      if(n.parent == null){
          return tmp;
      }
      
   if(!n.parent.splitAttr.isNumeric()){
      HashSet<String>values = new HashSet();
     if(n.parent.splitAttr.numValues() == 2){  
      for(int z = 0; z<n.parent.splitAttr.numValues();z++)
          values.add(n.parent.splitAttr.value(z).trim());
      
      if(values.contains("0") && values.contains("1")){
          if(n.parent.splitAttr.value(n.index).trim().equals("0"))
              tmp= "! "+"v"+n.parent.splitAttr.index();
          else tmp= "v"+n.parent.splitAttr.index();
      }
          else
             tmp= "v"+n.parent.splitAttr.index()+" = "+n.parent.splitAttr.value(n.index)+tmp;
     }
     else
        tmp= "v"+n.parent.splitAttr.index()+" = "+n.parent.splitAttr.value(n.index)+tmp;
   }
   else{
       if(n.index == 0){
           Double t = n.parent.splitValue;
           String spVal = "";
           String[] div = t.toString().split("\\.");
           int np = div[1].length();
           if(np == 0)
               spVal = String.format("%.1f",n.parent.splitValue);
           else spVal = spVal + n.parent.splitValue;
           tmp= "v"+n.parent.splitAttr.index()+" < "+spVal+tmp;
       }
       else{
          Double t = n.parent.splitValue-10e-6;
          String spVal = "";
          String[] div = t.toString().split("\\.");
           int np = div[1].length();
           if(np == 0)
               spVal = String.format("%.1f",n.parent.splitValue);
           else spVal = spVal + n.parent.splitValue;
          tmp= spVal+" < "+"v"+n.parent.splitAttr.index()+tmp; 
       }
   }
      
      
      if(n.parent.parent!=null)
                      tmp=" & "+tmp;
      
      tmp=createElConjunctionStringReReMiCodesHelp(n.parent)+tmp;
      
      return tmp;
      
  } 
    
    public String createElConjunctionReReMiCodesString(Node n){
      String tmp = "";
      int comp = n.complement;
      
    tmp = createElConjunctionStringReReMiCodesHelp(n);
      
      if(comp == 1){
          if(n.parent!=null)
              if(n.parent.parent!=null)
                    tmp = "! ( "+tmp+" )";
              else tmp = "! "+tmp;
          else tmp = "! "+tmp;
      }
      
      System.out.println("Create el. conjunction string");
      
      return tmp;
  } 
    
    public String createQueryReReMiCodesString(int queryIndex){
        String tmp = "";
        
        ArrayList<Node> t = redNodes.get(queryIndex);
        
        if(t!=null){
            if(t.size()>0){
                if(t.get(0)!=null){
                    if(t.get(0).parent!=null)
                        if(t.get(0).parent.parent!=null)
                            if(t.get(0).complement == 0)
                                 tmp = "( "+createElConjunctionReReMiCodesString(t.get(0))+" )";
                            else tmp = createElConjunctionReReMiCodesString(t.get(0));
                        else tmp = createElConjunctionReReMiCodesString(t.get(0));
                    else tmp = createElConjunctionReReMiCodesString(t.get(0));
                }
            }
        
        for(int i=1;i<t.size();i++)
            if(t.get(i)!=null){
                if(t.get(i).parent!=null)
                        if(t.get(i).parent.parent!=null)
                            if(t.get(i).complement == 0)
                                tmp+= " | "+"( "+createElConjunctionReReMiCodesString(t.get(i))+" )";
                            else tmp+= " | "+createElConjunctionReReMiCodesString(t.get(i));
                        else tmp+= " | "+createElConjunctionReReMiCodesString(t.get(i));
                else tmp+= " | "+createElConjunctionReReMiCodesString(t.get(i));
                //tmp+= " | "+ createElConjunctionReReMiString(t.get(i));
            }
        }
        
        return tmp;
    }
   
    public String createElConjunctionStringReReMiHelp(Node n){
      String tmp = "";
      
      if(n.parent == null){
          return tmp;
      }
      
   if(!n.parent.splitAttr.isNumeric()){
      HashSet<String>values = new HashSet();
     if(n.parent.splitAttr.numValues() == 2){  
      for(int z = 0; z<n.parent.splitAttr.numValues();z++)
          values.add(n.parent.splitAttr.value(z).trim());
      
      if(values.contains("0") && values.contains("1")){
          if(n.parent.splitAttr.value(n.index).trim().equals("0"))
              tmp= "! "+n.parent.splitAttr.name();
          else tmp= n.parent.splitAttr.name();
      }
          else
             tmp= n.parent.splitAttr.name()+" = "+n.parent.splitAttr.value(n.index)+tmp;
     }
     else
        tmp= n.parent.splitAttr.name()+" = "+n.parent.splitAttr.value(n.index)+tmp;
   }
   else{
       if(n.index == 0)
           tmp= n.parent.splitAttr.name()+" <= "+n.parent.splitValue+tmp;
       else
          tmp= n.parent.splitAttr.name()+" > "+n.parent.splitValue+tmp; 
   }
      
      
      if(n.parent.parent!=null)
                      tmp=" & "+tmp;
      
      tmp=createElConjunctionStringReReMiHelp(n.parent)+tmp;
      
      return tmp;
      
  } 
    
     public String createElConjunctionReReMiString(Node n){
      String tmp = "";
      int comp = n.complement;
      
    tmp = createElConjunctionStringReReMiHelp(n);
      
      if(comp == 1)
            tmp = "! ( "+tmp+" )";
      
      System.out.println("Create el. conjunction string");
      
      return tmp;
  } 
     
     
       public String createQueryReReMiString(int queryIndex){
        String tmp = "";
        
        ArrayList<Node> t = redNodes.get(queryIndex);
        
        if(t!=null){
            if(t.size()>0){
                if(t.get(0)!=null){
                    if(t.get(0).parent!=null)
                        if(t.get(0).parent.parent!=null)
                            if(t.get(0).complement == 0)
                                 tmp = "( "+createElConjunctionReReMiString(t.get(0))+" )";
                            else tmp = createElConjunctionReReMiString(t.get(0));
                        else tmp = createElConjunctionReReMiString(t.get(0));
                    else tmp = createElConjunctionReReMiString(t.get(0));
                }
            }
        
        for(int i=1;i<t.size();i++)
            if(t.get(i)!=null){
                if(t.get(i).parent!=null)
                        if(t.get(i).parent.parent!=null)
                            if(t.get(i).complement == 0)
                                tmp+= " | "+"( "+createElConjunctionReReMiString(t.get(i))+" )";
                            else tmp+= " | "+createElConjunctionReReMiString(t.get(i));
                        else tmp+= " | "+createElConjunctionReReMiString(t.get(i));
                else tmp+= " | "+createElConjunctionReReMiString(t.get(i));
                //tmp+= " | "+ createElConjunctionReReMiString(t.get(i));
            }
        }
        
        return tmp;
    }
    
  public String createElConjunctionStringHelp(Node n){
      String tmp = "";
      
      if(n.parent == null){
          return tmp;
      }
      
   if(!n.parent.splitAttr.isNumeric())
        tmp= n.parent.splitAttr.name()+"="+n.parent.splitAttr.value(n.index)+tmp;
   else{
       if(n.index == 0)
           tmp= n.parent.splitAttr.name()+"<="+n.parent.splitValue+tmp;
       else
          tmp= n.parent.splitAttr.name()+">"+n.parent.splitValue+tmp; 
   }
      
      
      if(n.parent.parent!=null)
                      tmp=" AND "+tmp;
      
      tmp=createElConjunctionStringHelp(n.parent)+tmp;
      
      return tmp;
      
  }  
    
  public String createElConjunctionString(Node n){
      String tmp = "";
      int comp = n.complement;
      
    tmp = createElConjunctionStringHelp(n);
      
      if(comp == 1)
            tmp = "NOT ("+tmp+")";
      
      System.out.println("Create el. conjunction string");
      
      return tmp;
  }
  
   public HashSet<Node> createElConjunctionInformationHelp(Node n){
      HashSet<Node> tmp = new HashSet<>();
      
      if(n.parent == null){
          return tmp;
      }
      
   tmp.add(n);
      
      
      tmp.addAll(createElConjunctionInformationHelp(n.parent));
      
      return tmp;
      
  }

 HashSet<Node> createElConjunctionInformation(Node n){
      
     HashSet<Node> tmp = new HashSet<>();
      int comp = n.complement;
      
    tmp = createElConjunctionInformationHelp(n);
    System.out.println("Num attributes in el. conjunction NEW: "+tmp.size());
      
      return tmp;
  }
  
    
//create arrays of node strings, one for positive, one for negated
//use them instead of this function - some change to nodes happen
    public String createElConjunctionString1(Node n){
        System.out.println("Create el. conjunction string");
        TwoTrees tmpT = new TwoTrees();
        String tmp = "";
        int comp = n.complement;
        Node ts = n.parent;
        int initindex = n.index, start = 1;
           int index = 0;
           
         ArrayList<Integer> indexes = new ArrayList<>();
         indexes.add(initindex);
           
        while(true){
            
            if(ts == null)
                break;

            if(start == 1){
                index = initindex;
                start = 0;
            }
          
            //n = n.parent;           
            tmp= ts.splitAttr.name()+"="+ts.splitAttr.value(index)+tmp;
            System.out.println("tmp: "+tmp);
            System.out.println("subree: ");
            index = ts.index;
            indexes.add(index);
		if(ts.parent!=null)
                      tmp=" AND "+tmp;
                
                if(ts.parent == null){
                    for(int i=indexes.size()-2;i>=1;i--)
                        ts = ts.children[indexes.get(i)];
                        n.parent = ts;
                    break;
                }
                else ts = ts.parent;
               
            }
        
        if(comp == 1)
            tmp = "NOT ("+tmp+")";
        
        return tmp;
        }
    
    public String createQueryString(int queryIndex){
        String tmp = "";
        
        ArrayList<Node> t = redNodes.get(queryIndex);
        
        if(t!=null){
            if(t.size()>0)
                if(t.get(0)!=null)
                 tmp = createElConjunctionString(t.get(0));
        
        for(int i=1;i<t.size();i++)
            if(t.get(i)!=null)
                tmp+= " OR "+ createElConjunctionString(t.get(i));
        }
        
        return tmp;
    }
    
    public void computeEntities(HashMap<Node,HashSet<Integer>> mapping){
        queryEntities.add(new HashSet<>());
        queryEntities.add(new HashSet<>());
        
        HashSet<Integer> tmp = new HashSet<>();
        for(Node n:this.redNodes.get(0)){
            if(n!=null)
               tmp.addAll(mapping.get(n));
        }
            
        queryEntities.get(0).addAll(tmp);
        
        tmp.clear();
        
        for(Node n:this.redNodes.get(1)){
            if(n!=null)
               tmp.addAll(mapping.get(n));
        }
            
        queryEntities.get(1).addAll(tmp);
        
    }
    
    public void normalizeRedescription(){
        for(int i=this.redNodes.get(0).size()-1;i>=0;i--){
            if(redNodes.get(0).get(i) == null)
                redNodes.get(0).remove(i);
        }
     
        for(int i=this.redNodes.get(1).size()-1;i>=0;i--){
            if(redNodes.get(1).get(i) == null)
                redNodes.get(1).remove(i);
        }
                
        HashSet<Integer> toRemove = new HashSet<>();
        
        //check and remove duplicate Nodes
        for(int i=0;i<this.redNodes.get(0).size();i++){
            String q1 = this.createElConjunctionString(this.redNodes.get(0).get(i));
            for(int j=i+1;j<this.redNodes.get(0).size();j++){
                String q2 = this.createElConjunctionString(this.redNodes.get(0).get(j));
                if(q1.trim().equals(q2.trim()))
                    toRemove.add(j);
            }
        }
        
        for(int i = this.redNodes.get(0).size()-1;i>=0;i--)
            if(toRemove.contains(i))
                 this.redNodes.get(0).remove(i);
        
        toRemove.clear();
        
        for(int i=0;i<this.redNodes.get(1).size();i++){
            String q1 = this.createElConjunctionString(this.redNodes.get(1).get(i));
            for(int j=i+1;j<this.redNodes.get(1).size();j++){
                String q2 = this.createElConjunctionString(this.redNodes.get(1).get(j));
                if(q1.trim().equals(q2.trim()))
                    toRemove.add(j);
            }
        }
        
        for(int i = this.redNodes.get(1).size()-1;i>=0;i--)
            if(toRemove.contains(i))
                 this.redNodes.get(1).remove(i);
        
    }
    
    
     double computePval(Instances dat){
        
         //assumes computeEntities function called before
         
         double pVal=1.0;
        int intersectSize = 0;
        
        for(int i:queryEntities.get(0))
                    if(queryEntities.get(1).contains(i))
                        intersectSize = intersectSize +1;
            
            double prob=((double)(queryEntities.get(0).size()*queryEntities.get(1).size()))/(dat.numInstances()*dat.numInstances());
            BinomialDistribution dist=new BinomialDistribution(dat.numInstances(),prob);
            pVal=1.0-dist.cumulativeProbability(intersectSize)+ dist.probability(intersectSize);
        
        return pVal;
    }
    
    public void computeStats(Instances data)
    {
        //compute all redescription statistics and measures
        //assumes computeEntities executed before
        // public double Jaccard, pval, mn;
        //public int support, supportUnion;
        //public ArrayList<Integer> querySupports;
        
        int sup=0, supU=0;
        
        for(int i:queryEntities.get(0))
            if(queryEntities.get(1).contains(i))
                sup++;
        
        supU = queryEntities.get(0).size()+queryEntities.get(1).size()-sup;
        
        support = sup;
        supportUnion = supU;
        
        Jaccard = ((double)support)/((double)supportUnion);
        
        if(supportUnion == 0)
            Jaccard = 0.0;
        
        int mnT = 0;
        
        for(int i=0;i<data.numInstances();i++)
            if(queryEntities.get(0).contains(i) && queryEntities.get(1).contains(i))
                    mnT++;
            else if(!queryEntities.get(0).contains(i) && !queryEntities.get(1).contains(i))
                mnT++;
        
        mn = mnT;
        pval = this.computePval(data);
        
        if(Double.isNaN(pval))
            pval = 1.0;
        
    }        
}
    
