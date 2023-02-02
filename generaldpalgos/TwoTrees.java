/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package generaldpalgos;


import java.math.BigDecimal;
import java.math.MathContext;
import java.math.RoundingMode;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Random;
import java.util.Set;
import org.apache.commons.math3.distribution.BinomialDistribution;
import org.apache.commons.math3.util.Pair;

import weka.core.*;


/**
 *
 * @author Matej Mihelcic, University of Eastern Finland implemented the dptressxpm.twotreesEM package as an extension of the 
 * original tree package decTreeWholeDP created for the manuscript: "Embedding differential privacy in decision tree
algorithm with different depths" by authors Xuanyu BAI , Jianguo YAO*, Mingxuan YUAN, Ke DENG,
 */

//This is the final version

public class TwoTrees{

	private static final long serialVersionUID = 1L;              

    public static final MathContext MATH_CONTEXT = new MathContext(20, RoundingMode.DOWN);  
    
	private int MaxDepth, minNodeSize;
	public void setMaxDepth(int maxDepth) {
		MaxDepth = maxDepth;
	}
        
        public void setminNodeSize(int minSize){
             minNodeSize = minSize;
        }
	
	private BigDecimal Epsilon = new BigDecimal(1.0);      
    public void setEpsilon(String epsilon) {                         
	    if (epsilon!=null && epsilon.length()!=0)
	        Epsilon = new BigDecimal(epsilon,MATH_CONTEXT);
    }
    
    private BigDecimal splitbudget = new BigDecimal(1.0,MATH_CONTEXT);
    private BigDecimal splitmarkovbudget = new BigDecimal(1.0,MATH_CONTEXT);
    private BigDecimal noisebudget = new BigDecimal(1.0,MATH_CONTEXT);
    
    private BigDecimal weightSplit = new BigDecimal(0.5,MATH_CONTEXT);
    private BigDecimal weightNoise = new BigDecimal(0.5,MATH_CONTEXT);
    
    public void setWeights(double w1, double w2){
        if(w1<0 || w2<0 || w1>1 || w2>1 || (w1+w1!=1.0))
            return;
        weightSplit = new BigDecimal(w1,MATH_CONTEXT);
        weightNoise = new BigDecimal(w2,MATH_CONTEXT);
    }
    
    public Instances view1Data = null;
    public Instances view2Data = null;
    
    private Random Random = new Random(); 
    public void setSeed(int seed){                          
		Random = new Random(seed);
	}
    
    private int MaxIteration, numLeafs = 0, MaxIterationRed, numClausesRed, maxSupport, minSupport, minSupportReal;                            
    
    public void setMaxIteration(int maxiteration){
    	MaxIteration = maxiteration;
    }
    
    public void setMaxIterationRed(int maxiteration){
        MaxIterationRed = maxiteration;
    }
    
    public void setNumClausesRed(int numClauses){
        numClausesRed = numClauses;
    }
    
    public void setMaxSupport(int sup){
        maxSupport = sup;
    }
    
    public void setMinSupport(int sup){
        minSupport = sup;
    }
    
    public void setMinSupportReal(int sup){
        minSupportReal = sup;
    }
    
    private double minimalJaccard, maxPval;
    
     public void setMinJaccard(double minJS){
        minimalJaccard = minJS;
    }
     
      public void setMaxPval(double pval){
        maxPval = pval;
    }
    
    private double EquilibriumThreshold;                 
    public void setEquilibriumThreshold(double equilibriumThreshold){
    	EquilibriumThreshold = equilibriumThreshold;
    }

    private double GiniSen = 1, rScoreSen = 1;       
    
	private Node RootFirst, RootSecond;
        public int numEntities = 0;
	
	private Set<Pair<Attribute, Double>> Attributes = new HashSet<Pair<Attribute,Double>>();
        private Set<Pair<Attribute, Double>> Attributes1 = new HashSet<Pair<Attribute, Double>>();
           
        public void setRootFirst(Node n){
            RootFirst = n;
        }
        
        public void setRootSecond(Node n){
            RootSecond = n;
        }
        
        public Instances addTargets(Instances data){
            Instances tmp = new Instances(data);
            int classIndex = data.classIndex();
            
            tmp.setClassIndex(-1);
            tmp.deleteAttributeAt(classIndex);
            int cl = countLeafs(RootFirst, 0);
        
            ArrayList<String> values1 = new ArrayList<>();
             // System.out.println("Num leaves: "+cl);
              for(int i=0;i<cl;i++)
                  values1.add("c"+i);
        
            tmp.insertAttributeAt(new Attribute("NewNominal",values1), tmp.numAttributes());
         
        HashSet<Integer> instInNode = new HashSet<>();
             
             for(int i=0;i<tmp.numInstances();i++)
                 instInNode.add(i);
             
             tmp.setClassIndex(tmp.numAttributes()-1);
             //System.out.println(toString(RootFirst));
             tmp = setTargets(RootFirst, classIndex, tmp, instInNode);
             
            return tmp;
        }

	private Instances[] partitionByAttr(Instances data, Attribute attr, double attrValue){
				
		Instances[] parts = null;
                
                if(attr.isNumeric())
                        parts = new Instances[2];
                 else       
                       parts =  new Instances[attr.numValues()];  
                
	    for(int i=0; i<parts.length; i++)
		{
			parts[i] = new Instances( data, data.numInstances() );         
		}
				
		Enumeration<Instance> instEnum = data.enumerateInstances(); 
		while(instEnum.hasMoreElements())                
		{
			Instance inst = instEnum.nextElement();
                        if(!attr.isNumeric()){
                            if(!inst.isMissing(attr))
                                  parts[(int)inst.value(attr)].add(inst);
                        }
                        else{
                            if(!inst.isMissing(attr)){
                                if(inst.value(attr)<=attrValue)
                                     parts[0].add(inst);
                                else parts[1].add(inst);
                            }
                        }
		}
				
		return parts;                                            
	}
	
	
    private void addNoiseDistribution(double[] count, BigDecimal budget){
    	                                       
    	int maxIndex = Utils.maxIndex(count);                  
    	
    	for(int i=0; i<count.length; i++)
    	{
    		count[i] += laplace(BigDecimal.ONE.divide(budget, MATH_CONTEXT));      
 
    		if(count[i] < 0)	                      
    			count[i] = 0;
    	}
 
    	double sum = Utils.sum(count);                           
    	if(sum <= 0){	                         
    		count[maxIndex] = 1;           
    	}
    } 

    private double laplace(BigDecimal bigBeta){                
    	
    	double miu = 0.;                                             
    	double beta=bigBeta.doubleValue();                
    	
        double uniform = Random.nextDouble()-0.5;            
        return miu-beta*((uniform>0) ? -Math.log(1.-2*uniform) : Math.log(1.+2*uniform));  
        
    }
    
  	
  	final int BufferSize = 500;   
  	double[] ScoreBuffer = new double[BufferSize];

  	int InitPointer = 0;           
  	int Pointer = 0;               
  	double variance = -1;                 
  	
  
        public int countLeafs(Node node, int count){
            
            if(node.isLeaf){
                count = count+1;
                return count;
            }
            
            for(int i=0;i<node.children.length;i++)
                count = countLeafs(node.children[i],count);
            
            return count;
        }
        
         void renumerateLeaves(Node n){
            if(n.parent==null){
                this.numLeafs = 0;
            }
            
            if(n.isLeaf){
                n.li = this.numLeafs;
                this.numLeafs++;
                return;
            }
            
           for(int i=0;i<n.children.length;i++) 
            renumerateLeaves(n.children[i]);
            
        }
        
         public Instances setTargets(Node node, int classIndex , Instances data, HashSet<Integer> instInNode){

            Attribute attr = node.splitAttr;
            
            if(node.isLeaf){
                //set targets and return
                for(int i:instInNode){
                    Instance inst = data.get(i);
                  //  System.out.println(inst.attribute(classIndex).name()+" "+classIndex);
                   // System.out.println("Node: "+"c"+node.li);
                    inst.setValue(classIndex, "c"+node.li);
                    data.set(i, inst);
                }
                
                return data;
            }
		
        if(!attr.isNumeric()){    
         for(int j=0;j<attr.numValues();j++){ 
             HashSet<Integer> instancesInANode = new HashSet<>();
            for(int i:instInNode)             
		{                     
			Instance inst = data.get(i); 
                        if(inst.isMissing(attr))
                            inst.setClassMissing();
                        else if(inst.stringValue(attr).trim().equals(attr.value(j).trim()))//prije preko double-a
                            instancesInANode.add(i);
		}
            data = setTargets(node.children[j], classIndex, data,  instancesInANode);
         }
       }
        else{
            for(int j=0;j<2;j++){
                HashSet<Integer> instancesInANode = new HashSet<>();
            for(int i:instInNode)             
		{
                     //handle missing values
			Instance inst = data.get(i); 
                        if(inst.isMissing(attr))//it is missing
                            inst.setClassMissing();                           
                        else if(j==0 && inst.value(attr)<=node.splitValue)
                            instancesInANode.add(i);
                        else if(j==1 && inst.value(attr)>node.splitValue)
                            instancesInANode.add(i);
		}
            data = setTargets(node.children[j], classIndex, data,  instancesInANode);
            }
        }
            
            return data;
        }
         
         public Node returnRoot(Node n){
             
             if(n.parent == null)
                 return n;
             
           return  returnRoot(n.parent);
              
         }
         
         HashSet<Integer> computeLeafEntityIDs(Node n, Instances data, HashSet<Integer> instancesInNode){
             HashSet<Integer> tmp = new HashSet<>();
            
             if(n.parent == null){
                 tmp.addAll(instancesInNode);
                 return tmp;
             }
             
              Attribute attr = n.parent.splitAttr;
            
             HashSet<Integer> toRemove = new HashSet<>(instancesInNode.size());
            for(int i:instancesInNode)             
		{
                     
			Instance inst = data.get(i); 
                        if(!attr.isNumeric()){
                            if(inst.isMissing(attr))
                                toRemove.add(i);
                            else if(!inst.stringValue(attr).trim().equals(attr.value(n.index).trim()))
                                toRemove.add(i);
                        }
                        else{
                            if(inst.isMissing(attr))
                                toRemove.add(i);
                            else if((n.index == 0) && (inst.value(attr)>n.parent.splitValue)){
                                toRemove.add(i);
                            }
                            else if((n.index == 1) && (inst.value(attr)<=n.parent.splitValue)){
                                toRemove.add(i);
                            }
                        }                     
		}
                instancesInNode.removeAll(toRemove);
                tmp = computeLeafEntityIDs(n.parent, data, instancesInNode);
                        //     n = n.children[indexes.get(count)];
             
             return tmp;
         }
         
         //must be changed
         HashSet<Integer> computeLeafEntityIDs1(Node n, Instances data, HashSet<Integer> instancesInNode){
             HashSet<Integer> tmp = new HashSet<>();
             ArrayList<Integer> indexes = new ArrayList<>();
             
             while(n.parent!=null){
                 indexes.add(n.index);
                 n=n.parent;
             }
             
             int count =0;
             while(!n.isLeaf){
                 
                 Attribute attr = n.splitAttr;
            
             HashSet<Integer> toRemove = new HashSet<>(instancesInNode.size());
            for(int i:instancesInNode)             
		{
                     
			Instance inst = data.get(i); 
                        if(inst.value(attr) !=  Double.parseDouble(attr.value(indexes.get(count))))
                            toRemove.add(i);
		}
                instancesInNode.removeAll(toRemove);
                             n = n.children[indexes.get(count)];
                             count++;

             }

                     tmp.addAll(instancesInNode);
                                  
             return tmp;
         }
         
         public Instances copyTargets(Instances data1, Instances data2){
             Instances tmp = new Instances(data2);
             
             int classIndex = tmp.classIndex();
             
             if(classIndex>=0){
                 tmp.setClassIndex(-1);
                 tmp.deleteAttributeAt(classIndex);       
             }
               
             
             ArrayList<String> values = new ArrayList<>();
             
             for (int i=0;i<data1.classAttribute().numValues();i++) {
                values.add(data1.classAttribute().value(i));
             }
             
             int ind = tmp.numAttributes();
             tmp.insertAttributeAt(new Attribute("NewNominal",values), tmp.numAttributes());
             tmp.setClassIndex(ind);
             
             for(int i=0;i<data1.numInstances();i++)
                 if(!data1.get(i).classIsMissing())
                     tmp.get(i).setValue(ind, data1.get(i).stringValue(data1.classIndex()));
                 else tmp.get(i).setClassMissing();
                //  else 
             
             
             return tmp;
         }
	
    public void updateTargetDistribution(Node node, Instances data){
                        node.data = data;
			 
			double[] tempcount = new double[data.numClasses()];                
			Enumeration<Instance> instEnum = data.enumerateInstances();  
			while(instEnum.hasMoreElements()){                             
				Instance inst = (Instance)instEnum.nextElement(); 
                                if(!inst.classIsMissing())
                                    tempcount[(int)inst.classValue()]++;  
			}
				
			node.count = tempcount;                                   
			node.dist = node.count.clone();                      
			if(Utils.sum(node.dist) != 0.0)                       
				Utils.normalize(node.dist); 
                        
                  if(node.isLeaf)
                            return; 
                        
                Instances[] parts = partitionByAttr(node.data, node.splitAttr, node.splitValue);                 
		
		for(int i=0; i < parts.length; i++){
                    updateTargetDistribution(node.children[i], parts[i]);
		}        
                        
    }
    
    public HashMap<Node,HashSet<Integer>> getLeafSupports(Node node, Instances data, HashSet<Integer> instInNode){
        HashMap<Node,HashSet<Integer>> tmp = new HashMap<>();
        
         Attribute attr = node.splitAttr;
            
            if(node.isLeaf){
                if(!tmp.containsKey(node))
                    tmp.put(node, new HashSet<>());
                //set targets and return
                for(int i:instInNode){
                    tmp.get(node).add(i);
                }
                
                return tmp;
            }
		
            
         for(int j=0;j<attr.numValues();j++){ 
             HashSet<Integer> instancesInANode = new HashSet<>();
            for(int i:instInNode)             
		{
                     
			Instance inst = data.get(i); 
                        if(inst.value(attr) ==  Double.parseDouble(attr.value(j)))
                            instancesInANode.add(i);
		}
            tmp = getLeafSupports(node.children[j], data,  instancesInANode);
         }
        
        return tmp;
    }
         
    public Node getFirstRoot(){
        return RootFirst;
    }   
    
    Redescription copyRedescriptionNodes(Redescription orig){
        Redescription tmp = new Redescription();
        
        for(int i=0;i<orig.redNodes.get(0).size();i++){
            //Node n = new Node(orig.redNodes.get(0).get(i));
            tmp.redNodes.get(0).add(orig.redNodes.get(0).get(i));
        }
        
        for(int i=0;i<orig.redNodes.get(1).size();i++){
            //Node n = new Node(orig.redNodes.get(1).get(i));
            tmp.redNodes.get(1).add(orig.redNodes.get(1).get(i));
        }
        
        return tmp;
    }
    
    public ArrayList<Node> getLeafs(int tree, Node n, ArrayList<Node> tmp){
        
        if(n.isLeaf){
           // if(n.data.numInstances()>minSupportReal)//can not have this condition
                tmp.add(n);
                return tmp;
            }

      if(!n.splitAttr.isNumeric()){  
        for(int i=0;i<n.splitAttr.numValues();i++){
            Node t = n.children[i];
            
            tmp = getLeafs(tree,t, tmp);
        }
      }
      else{
          for(int i=0;i<2;i++){
               Node t = n.children[i];            
               tmp = getLeafs(tree,t, tmp);
          }
              
      }
        
        return tmp;
    }
    
    public int leafMatches(Instances data, Instances data1){
    
        ArrayList<Node> first = getLeafs(0, RootFirst, new ArrayList<Node>());
        ArrayList<Node> second = getLeafs(1, RootSecond, new ArrayList<Node>());
        
        DecTreeExpMech phony = new DecTreeExpMech();
        
        int numOK = 0;
        
        int size = first.size();
        for(int i=0;i<size;i++){
            first.add(new Node(first.get(i)));
            first.get(first.size()-1).complement = 1;
            first.get(first.size()-1).parent = first.get(i).parent;
        }
        
        size = second.size();
        for(int i=0;i<size;i++){
            second.add(new Node(second.get(i)));
            second.get(second.size()-1).complement = 1;
            second.get(second.size()-1).parent = second.get(i).parent;
        }      
        
        //compute supports of all nodes and store in a HashMap
        HashMap<Node, HashSet<Integer>> map = new HashMap<>();
        
        HashSet<Integer> instances = new HashSet<>();
        HashSet<Integer> instancesInNode = new HashSet<>();
        
        for(int i=0;i<data.numInstances();i++)
            instances.add(i);
        
        for(int i=0;i<first.size();i++){
            if(first.get(i).complement==0){
            instancesInNode.clear();
            instancesInNode.addAll(instances);
            map.put(first.get(i), this.computeLeafEntityIDs(first.get(i), data, instancesInNode));
            }
            else{
                HashSet<Integer> ents = new HashSet<>();
                for(int in:instances){
                    if(!map.get(first.get(i-first.size()/2)).contains(in))
                        ents.add(in);
                }
                map.put(first.get(i),ents);
            }
        }
        
        
        for(int i=0;i<second.size();i++){
            if(second.get(i).complement==0){
            instancesInNode.clear();
            instancesInNode.addAll(instances);
            map.put(second.get(i), this.computeLeafEntityIDs(second.get(i), data1, instancesInNode));
            }
            else{
                HashSet<Integer> ents = new HashSet<>();
                for(int in:instances){
                    if(!map.get(second.get(i-second.size()/2)).contains(in))
                        ents.add(in);
                }
                map.put(second.get(i),ents);
            }
        }
        
        double union = 0.0, intersection = 0.0;
        
        System.out.println("first size: "+first.size());
        System.out.println("second size: "+second.size());
        
        ArrayList<Double> jaccards = new ArrayList<>();
        ArrayList<String> q1 = new ArrayList<>();
        ArrayList<String> q2 = new ArrayList<>();
        ArrayList<Integer> sp = new ArrayList<>();
        ArrayList<Double> pval = new ArrayList<>();
        Redescription rt = new Redescription();
        
        for(int i=0;i<first.size();i++){
            HashSet<Integer> ents1 = map.get(first.get(i));
            
            union = 0.0; intersection = 0.0;
            
            q1.add(rt.createElConjunctionString(first.get(i)));
        
            for(int j=0;j<second.size();j++){
                HashSet<Integer> ents2 = map.get(second.get(j));
                if(q2.size() == j)
                     q2.add(rt.createElConjunctionString(second.get(j)));
                for(int k:ents1)
                    if(ents2.contains(k))
                        intersection = intersection +1;
                   if(ents1.size() == 0 || ents2.size() == 0){
                       System.out.println("Empty node!");
                       System.out.println(rt.createElConjunctionString(first.get(i)));
                       System.out.println(rt.createElConjunctionString(second.get(j)));
                       continue;
                   }
                    jaccards.add((intersection/(ents1.size()+ents2.size()-intersection)));
                    sp.add((int)intersection);
                    rt.queryEntities.add(ents1);
                    rt.queryEntities.add(ents2);
                    pval.add(rt.computePval(data));
                    rt.queryEntities.clear();
                    intersection = 0;
            }
            
        }
        
         //Collections.sort(jaccards, Collections.reverseOrder()); 
         System.out.println(); System.out.println();
         System.out.println("Redescriptions: ");
         System.out.println();
         int c1=0, c2=0;
         for(int i=0;i<jaccards.size();i++){
             if(jaccards.get(i)<this.minimalJaccard)
                 continue;
             if(pval.get(i)>=this.maxPval)
                 continue;
             if(sp.get(i) > this.maxSupport || sp.get(i)<this.minSupport)
                 continue;
             if(c2<second.size()){
                System.out.println(q1.get(c1));
                System.out.println(q2.get(c2++));
             }
             else{
                 c2 = 0;
                 c1++;
                System.out.println(q1.get(c1));
                System.out.println(q2.get(c2++));     
             }
             numOK++;
             System.out.println(jaccards.get(i)+" "+sp.get(i)+" "+pval.get(i));
             System.out.println();
         }
        return numOK;
    }
    
    public Redescription nosyStatistics(Redescription r, Instances data, Instances data1){
        
       // count[i] += laplace(BigDecimal.ONE.divide(budget, MATH_CONTEXT));  
       
       int intersection = r.support;
       int intersectionComplement = data.numInstances()-r.support;
       int q1Supp = r.queryEntities.get(0).size();
       int q2Supp = r.queryEntities.get(1).size();
       
       BigDecimal budg = this.noisebudget.divide(new BigDecimal(3.0));
       
       intersection += laplace(BigDecimal.ONE.divide(budg, MATH_CONTEXT)); 
       intersectionComplement += laplace(BigDecimal.ONE.divide(budg, MATH_CONTEXT)); 
       q1Supp+= laplace(BigDecimal.ONE.divide(budg, MATH_CONTEXT));  
       q2Supp+= laplace(BigDecimal.ONE.divide(budg, MATH_CONTEXT));  
       
       System.out.println("intersection: "+r.support+" "+intersection);
       System.out.println("intersectionComplement: "+intersectionComplement);
       System.out.println("q1Supp: "+r.queryEntities.get(0).size()+" "+q1Supp);
       System.out.println("q2Supp: "+r.queryEntities.get(1).size()+" "+q2Supp);
       
       int numEnt = intersection + intersectionComplement;
       
       
       
       if(intersectionComplement<0)
           intersectionComplement = 0;
       
           if(q1Supp> numEnt)
           q1Supp = numEnt;
       if(q2Supp>numEnt)
           q2Supp = numEnt;
       
       if(q1Supp<0)
           q1Supp = 0;
       if(q2Supp <0)
           q2Supp = 0;
       
       if(intersection<0)
           intersection = 0;
       
       int union = q1Supp + q2Supp - intersection;
       
       if(union < 0)
           union = 0;
       
       if(union > numEnt)
           union = numEnt;
       
       r.Jaccard = intersection/(double)union;
       
       if(r.Jaccard>1.0)
           r.Jaccard = 1.0;
       
       if(r.Jaccard<0.0)
           r.Jaccard = 0.0;

        double prob=((double)(q1Supp*q2Supp))/(numEnt*numEnt);
        if(prob>1.0)
            prob = 1.0;
        if(prob<0.0)
            prob = 0.0;
        
        BinomialDistribution dist=new BinomialDistribution(numEnt,prob);
        double pVal=1.0-dist.cumulativeProbability(intersection);
        
         double probReal=((double)(r.queryEntities.get(0).size()*r.queryEntities.get(1).size()))/(data.numInstances()*data.numInstances());
            BinomialDistribution dist1=new BinomialDistribution(data.numInstances(),probReal);
            double pValReal=1.0-dist1.cumulativeProbability(r.support);
            
            System.out.println("probs: "+probReal+" "+prob);
            System.out.println("pvals: "+pValReal+" "+pVal);
       
       r.pval = pVal ;
       r.support = intersection;
       r.supportUnion = union;
       
        return r;
    }
    
    //statistic that uses two passess on the database to compute the red. statistics
     public Redescription nosyStatistics1(Redescription r, Instances data, Instances data1){

       int intersection = r.support;
       int q1Supp = r.queryEntities.get(0).size();
       int q2Supp = r.queryEntities.get(1).size();
       int q1Minusq2Supp = q1Supp - intersection;
       int remainingPart = data.numInstances() - intersection - q1Minusq2Supp;
       
       //always apply the whole query nosisy count for the smaller query
       
       
       
       BigDecimal budg = this.noisebudget.divide(new BigDecimal(2.0));
       
       intersection += laplace(BigDecimal.ONE.divide(budg, MATH_CONTEXT)); 
       q1Minusq2Supp+= laplace(BigDecimal.ONE.divide(budg, MATH_CONTEXT)); 
       remainingPart +=laplace(BigDecimal.ONE.divide(budg, MATH_CONTEXT));
       //intersectionComplement += laplace(BigDecimal.ONE.divide(budg, MATH_CONTEXT)); 
       q1Supp = q1Minusq2Supp + intersection; 
       q2Supp+= laplace(BigDecimal.ONE.divide(budg, MATH_CONTEXT));  
       
       System.out.println("intersection: "+r.support+" "+intersection);
       System.out.println("q1Supp: "+r.queryEntities.get(0).size()+" "+q1Supp);
       System.out.println("q2Supp: "+r.queryEntities.get(1).size()+" "+q2Supp);
       
       if(q1Minusq2Supp<0)
           q1Minusq2Supp = 0;
       
       int numEnt = intersection + q1Minusq2Supp + remainingPart;
       
       int union = q1Supp + q2Supp - intersection;
       
       if(remainingPart<0)
           remainingPart = 0;
       
           if(q1Supp> numEnt)
           q1Supp = numEnt;
           
       if(q2Supp>numEnt)
           q2Supp = numEnt;
       
       if(q1Supp<0)
           q1Supp = 0;
       if(q2Supp <0)
           q2Supp = 0;
       
       if(intersection<0)
           intersection = 0;
       
       if(union < 0)
           union = 0;
       
       if(union > numEnt)
           union = numEnt;
       
       r.Jaccard = intersection/(double)union;
       
       if(r.Jaccard>1.0)
           r.Jaccard = 1.0;
       
       if(r.Jaccard<0.0)
           r.Jaccard = 0.0;

        double prob=((double)(q1Supp*q2Supp))/(numEnt*numEnt);
        if(prob>1.0)
            prob = 1.0;
        if(prob<0.0)
            prob = 0.0;
        
        BinomialDistribution dist=new BinomialDistribution(numEnt,prob);
        double pVal=1.0-dist.cumulativeProbability(intersection);
        
         double probReal=((double)(r.queryEntities.get(0).size()*r.queryEntities.get(1).size()))/(data.numInstances()*data.numInstances());
            BinomialDistribution dist1=new BinomialDistribution(data.numInstances(),probReal);
            double pValReal=1.0-dist1.cumulativeProbability(r.support);
            
            System.out.println("probs: "+probReal+" "+prob);
            System.out.println("pvals: "+pValReal+" "+pVal);
       
       r.pval = pVal ;
       r.support = intersection;
       r.supportUnion = union;
       
        return r;
    }
    
    
    //create a function that computes redescription measures using noisy count on query supports
    ArrayList<Double> evalExpansion(HashSet<Integer> q1, HashSet<Integer> q2, ArrayList<Integer> q1Supps, ArrayList<Integer> q2Supps , HashMap<Integer,HashMap<Integer,Integer>> intersections, int numEntities){
        ArrayList<Double> result = new ArrayList<>(6);
        //0-q1Supp, 1-q2Supp, 2-intersection, 3-union, 4-JS, 5-pval
        
        HashSet<Integer> posActive = new HashSet<>();
        int q1Supp = 0;
        
        int ctmp = 0;
        
        for(int i:q1){
            if(q1.contains(i) && q1.contains(-i) && i>0){
                System.out.println("Tautology...");
                    for(int z:q1)
                        System.out.print(z+" ");
                    System.out.println();
            }
            if(i>=0)
                posActive.add(i-1);
            else{
                for(int j=0;j<q1Supps.size();j++){
                    if(j!=(Math.abs(i)-1))
                        posActive.add(j);
                }
            }
        }
        
        for(int i:posActive)
            q1Supp+=q1Supps.get(i);
        
        System.out.println("q1 supp count: "+q1Supp);
        for(int i:posActive)
            System.out.print(i+" ");
        System.out.println();
        
        
       
        
        HashSet<Integer> posActiveQ2 = new HashSet<>();
        int q2Supp = 0;
        
        for(int i:q2){
             if(q2.contains(i) && q2.contains(-i) && i>0){
                System.out.println("Tautology...");
             }
            if(i>=0)
                posActiveQ2.add(i-1);
            else{
                for(int j=0;j<q2Supps.size();j++)
                    if(j!=(Math.abs(i)-1))
                        posActiveQ2.add(j);
            }
        }
        
        System.out.println("Num entities in: "+numEntities);
        
        for(int i:posActiveQ2)
            q2Supp+=q2Supps.get(i);
        
        System.out.println("q2 supp count: "+q2Supp);
        for(int i:posActiveQ2)
            System.out.print(i+" ");
        System.out.println();
        
        
        if(numEntities<=0)
            numEntities = Math.max(q1Supp, q2Supp);
        
        if(numEntities<=0)
            numEntities = 0;
        
         if(q1Supp>numEntities){
           System.out.println("q1 to large...");
            q1Supp = numEntities;
        }
       
        if(q2Supp>numEntities){
             System.out.println("q2 to large...");
            q2Supp = numEntities;
        }
        
        int intersection = 0;
        
        for(int i:posActive)
            for(int j:posActiveQ2){
                intersection+=intersections.get(i).get(j);
            }
        
        System.out.println("Intersection size: "+intersection);
        
         if(q1Supp<0)
            q1Supp = 0;
        if(q2Supp < 0)
            q2Supp = 0;
        if(intersection<0)
            intersection = 0;

        if(intersection>numEntities){
             System.out.println("intersection to large... "+intersection+" "+numEntities);
            intersection = numEntities;
        }
        
        if(intersection>q1Supp || intersection>q2Supp)
            intersection = Math.min(q1Supp, q2Supp);
        
        int union = q1Supp + q2Supp - intersection;

        if(union>numEntities){
            System.out.println("union to large... "+union+" "+numEntities);
            union = numEntities;
        }
        
        if(union<0)
            union = 0;
        
        
            if(q1Supp>union)
            union = q1Supp;
        if(q2Supp>union)
            union = q2Supp;
        
        if(intersection>union){
            System.out.println("intersection>union... "+intersection+" "+union);
            union = intersection;
        }
        
    

        double JS = ((double) intersection)/((double)union);
        
        if(JS>1.0)
            JS = 1.0;
        
        double pval = 0;
        
         double prob=((double)(q1Supp*q2Supp))/(numEntities*numEntities);
         if(prob>1.0)
             prob = 1.0;
         if(prob<0.0)
             prob = 0.0;
         BinomialDistribution dist1=new BinomialDistribution(numEntities,prob);
            pval=1.0-dist1.cumulativeProbability(intersection);
       
         result.add((double)q1Supp); result.add((double)q2Supp); 
         result.add((double) intersection); result.add((double)union);
         result.add(JS); result.add(pval);
            
        return result;
    }
     
    //auxiliary function that creates a redescription
    Redescription createRed(HashSet<Integer> q1Ind, HashSet<Integer> q2Ind, ArrayList<Node> first, ArrayList<Node> second, ArrayList<Double> stats){
        Redescription tmp = new Redescription();
        DecTreeExpMech phony = new DecTreeExpMech();

        tmp.querySupports.add((int)stats.get(0).doubleValue());
        tmp.querySupports.add((int)stats.get(1).doubleValue());
        tmp.support = (int) stats.get(2).doubleValue();
        tmp.supportUnion = (int)stats.get(3).doubleValue();
        tmp.Jaccard = stats.get(4);
        tmp.pval = stats.get(5);

        for(int i:q1Ind){
            Node t = new Node();
            t.copyNodeFull(first.get(Math.abs(i)-1));
            if(i<0)
                t.complement = 1;
            
            tmp.redNodes.get(0).add(t);
        }

        for(int i:q2Ind){
            Node t = new Node();
            t.copyNodeFull(second.get(Math.abs(i)-1));
            if(i<0)
                t.complement = 1;
            tmp.redNodes.get(1).add(t);
        }
        
        return tmp;
    }
     
    //create a function that creates redescriptions using noisy count
     
     public ArrayList<Redescription> createRedescriptionsNoisy(Instances data, Instances data1, int maxClauses, int flip){
         ArrayList<Redescription> result = new ArrayList<>();
         
         //save all the leafs in two arrays 
         
         System.out.println("First tree: ");
         System.out.println(toString(RootFirst));
         
         
          System.out.println("Second tree: ");
         System.out.println(toString(RootSecond));
         
           ArrayList<Node> first = getLeafs(0, RootFirst, new ArrayList<Node>());
        ArrayList<Node> second = getLeafs(1, RootSecond, new ArrayList<Node>());
        
        //compute supports of all nodes and store in a HashMap
        HashMap<Node, HashSet<Integer>> map = new HashMap<>();
        
        HashSet<Integer> instances = new HashSet<>();
        HashSet<Integer> instancesInNode = new HashSet<>();
        
        for(int i=0;i<data.numInstances();i++)
            instances.add(i);
        
        for(int i=0;i<first.size();i++){
            if(first.get(i).complement==0){
            instancesInNode.clear();
            instancesInNode.addAll(instances);
            map.put(first.get(i), this.computeLeafEntityIDs(first.get(i), data, instancesInNode));
            }
            else{//never computed
                HashSet<Integer> ents = new HashSet<>();
                for(int in:instances){
                    if(!map.get(first.get(i-first.size()/2)).contains(in))
                        ents.add(in);
                }
                map.put(first.get(i),ents);
            }
        }
        
        
        for(int i=0;i<second.size();i++){
            if(second.get(i).complement==0){
            instancesInNode.clear();
            instancesInNode.addAll(instances);
            map.put(second.get(i), this.computeLeafEntityIDs(second.get(i), data1, instancesInNode));
            }
            else{//never computed
                HashSet<Integer> ents = new HashSet<>();
                for(int in:instances){
                    if(!map.get(second.get(i-second.size()/2)).contains(in))
                        ents.add(in);
                }
                map.put(second.get(i),ents);
            }
        }

          BigDecimal budg = this.Epsilon.divide(new BigDecimal(2.0));
           System.out.println("Reds budget: "+budg);
      
        //compute the noisy intersections + q1 supports of each query with every query from W2
        HashMap<Integer, HashMap<Integer,Integer>> supportIntersection = new HashMap<>();
         ArrayList<Integer> reminders = new ArrayList<>();//with the second(0)
        
        for(int i=0;i<first.size();i++){
            HashSet<Integer> entities1 = map.get(first.get(i));
            for(int j=0;j<second.size();j++){
                int intersection = 0;
                HashSet<Integer> entities2 = map.get(second.get(j));
                
                for(int k:entities1)
                    if(entities2.contains(k))
                        intersection++;
                
                intersection+=laplace(BigDecimal.ONE.divide(budg, MATH_CONTEXT));
                
                if(!supportIntersection.containsKey(i)){
                    supportIntersection.put(i, new HashMap<>());
                }
                
                supportIntersection.get(i).put(j, intersection);
                
                if(j == 0){
                    int reminder = entities1.size() - intersection;
                    reminder += laplace(BigDecimal.ONE.divide(budg, MATH_CONTEXT));
                    reminders.add(reminder);
                }          
            }
        }
        
        
         ArrayList<Integer> supportsT2 = new ArrayList<>();
         
         System.out.println("Entities2 size: ");
         //compute the noisy count of q2 supports
         for(int i=0;i<second.size();i++){
             HashSet<Integer> entities2 = map.get(second.get(i));
             System.out.print(entities2.size()+" ");
             int sup = entities2.size();
             sup+=  laplace(BigDecimal.ONE.divide(budg, MATH_CONTEXT));
             supportsT2.add(sup);
         }
         
         System.out.println();
         
         ArrayList<Integer> supportsUnion = new ArrayList<>();
         
         //compute supports T1
         ArrayList<Integer> supportsT1 = new ArrayList<>();
         
         for(int i=0;i<reminders.size();i++){
             supportsT1.add(reminders.get(i)+supportIntersection.get(i).get(0));
         }

         //compute and create redescriptions (including negations and disjunctions), all reds containing 
         //predefined maximal number of clauses (numClausesRed)
         HashSet<Integer> usedW1 = new HashSet<>();
         HashSet<Integer> usedW2 = new HashSet<>();
         
         //implement a greedy procedure for creating redescriptions
            //a) first pair one node from first to one from second
            //b) add disjunctive node that increases JS the most keeping p in the given bounds
         
             double oldJS = 0.0, newJS = 0.0;
             int numEntities = 0;
             
             for(int i=0;i<supportsT2.size();i++)
                 numEntities+= Math.max(0,supportsT2.get(i));
             
             System.out.println("Supports T2size: "+supportsT2.size());
             for(int cc=0;cc<supportsT2.size();cc++)
                 System.out.print(supportsT2.get(cc)+" ");
             System.out.println();
             System.out.println("Num entities main: "+numEntities);
             
             // ArrayList<Double> evalExpansion(HashSet<Integer> q1, HashSet<Integer> q2, ArrayList<Integer> q1Supps, ArrayList<Integer> q2Supps , HashMap<Integer,HashMap<Integer,Integer>> intersections, int numEntities)
            
             int selection=-1;
             
            for(int i=0;i<first.size();i++){
                for(int j=0;j<second.size();j++){
                     ArrayList<Double> statsFin = new ArrayList<>();
                     Redescription tmpRed = null;
                    for(int c=0;c<4;c++){
                        if(c == 0){
                             usedW1.add(i+1); usedW2.add(j+1);
                        }
                        else if(c==1){
                                usedW1.add(-(i+1)); usedW2.add(j+1);
                                }
                        else if(c==2){
                            usedW1.add(i+1); usedW2.add(-(j+1));
                        }
                        else if(c==3){
                            usedW1.add(-(i+1)); usedW2.add(-(j+1));
                        }
                        //select the pair with best JS
                        
                        ArrayList<Double> stats = evalExpansion(usedW1, usedW2,  supportsT1, supportsT2 , supportIntersection, numEntities);
                        
                        if(c == 0){
                            selection = 0;
                            statsFin.addAll(stats);
                            tmpRed = createRed(usedW1, usedW2, first, second, stats );
                        }
                        else{
                            if(stats.get(4)>tmpRed.Jaccard && stats.get(2)<=this.maxSupport){
                                if(tmpRed.pval<this.maxPval){
                                    if(stats.get(5)<this.maxPval){
                                        tmpRed = createRed(usedW1, usedW2, first, second, stats );
                                        statsFin.clear();
                                        statsFin.addAll(stats);
                                        selection = c;
                                    }
                                }
                                else{
                                    tmpRed = createRed(usedW1, usedW2, first, second, stats );
                                    statsFin.clear();
                                    statsFin.addAll(stats);
                                    selection = c;
                                }
                            }
                            else{
                                if(tmpRed.pval>this.maxPval && stats.get(2)<=this.maxSupport){
                                    if(stats.get(5)<this.maxPval){
                                         tmpRed = createRed(usedW1, usedW2, first, second, stats );
                                         statsFin.clear();
                                         statsFin.addAll(stats);
                                         selection = c;
                                    }
                                }
                            }
                        }
                        usedW1.clear(); usedW2.clear();
                    }
                    
                    if(tmpRed.support>this.maxSupport){
                        usedW1.clear(); usedW2.clear();
                        continue;
                    }
                    
                    usedW1.clear(); usedW2.clear();
                    
                    if(selection == 0){
                        usedW1.add(i+1); usedW2.add(j+1);
                    }
                    else if(selection == 1){
                        usedW1.add(-(i+1)); usedW2.add(j+1);
                    }
                    else if(selection == 2){
                        usedW1.add(i+1); usedW2.add(-(j+1));
                    }
                    else if(selection == 3){
                        usedW1.add(-(i+1)); usedW2.add(-(j+1));
                    }
                          
                    //create a redescription
                    for(int k=1;k<maxClauses;k++){
                        int maxInd = -1, negated = 0;
                        double JSDiff = Double.NEGATIVE_INFINITY;
                        //add clause to q1 if it improves JS with pv<maxP
                        for(int w1=i+1;w1<supportsT1.size();w1++){
                            
                            if(usedW1.contains(w1+1) || usedW1.contains(-(w1+1)))
                                continue;
                            
                            //try the positive and negated clause
                             usedW1.add(w1+1);
                             ArrayList<Double> stats = evalExpansion(usedW1, usedW2,  supportsT1, supportsT2 , supportIntersection, numEntities);           
                            
                              if(stats.get(4)>tmpRed.Jaccard && stats.get(2)<=this.maxSupport){
                                if(tmpRed.pval<this.maxPval){
                                    if(stats.get(5)<this.maxPval && (stats.get(4)-tmpRed.Jaccard)>JSDiff){
                                        maxInd = w1+1;
                                        JSDiff = stats.get(4)-tmpRed.Jaccard;
                                    }
                                }
                                else{
                                    if((stats.get(4)-tmpRed.Jaccard)>JSDiff && stats.get(2)<=this.maxSupport){
                                            maxInd = w1+1;
                                            JSDiff = stats.get(4)-tmpRed.Jaccard;
                                    }
                                }
                            }
                            else{
                                if(tmpRed.pval>this.maxPval && stats.get(2)<=this.maxSupport){
                                    if(stats.get(5)<this.maxPval){
                                         maxInd = w1+1;
                                         JSDiff = stats.get(4)-tmpRed.Jaccard;
                                    }
                                }
                            }
                              
                              usedW1.remove(w1+1);
                              
                              //negated
                              
                               usedW1.add(-(w1+1));
                             stats = evalExpansion(usedW1, usedW2,  supportsT1, supportsT2 , supportIntersection, numEntities);           
                            
                              if(stats.get(4)>tmpRed.Jaccard && stats.get(2)<=0.8*numEntities/*this.maxSupport*/){
                                if(tmpRed.pval<this.maxPval){
                                    if(stats.get(5)<this.maxPval && (stats.get(4)-tmpRed.Jaccard)>JSDiff){
                                        maxInd = w1+1;
                                        JSDiff = stats.get(4)-tmpRed.Jaccard;
                                        negated = 1;
                                    }
                                }
                                else{
                                    if((stats.get(4)-tmpRed.Jaccard)>JSDiff){
                                            maxInd = w1+1;
                                            JSDiff = stats.get(4)-tmpRed.Jaccard;
                                            negated = 1;
                                    }
                                }
                            }
                            else{
                                if(tmpRed.pval>this.maxPval && stats.get(2)<=this.maxSupport){
                                    if(stats.get(5)<this.maxPval && stats.get(3)<this.maxSupport){
                                         maxInd = w1+1;
                                         JSDiff = stats.get(4)-tmpRed.Jaccard;
                                         negated = 1;
                                    }
                                }
                            }
                              
                              usedW1.remove(-(w1+1));  
                        }
                        
                        //create final expansion
                        if(maxInd>-1){
                            if(negated == 1)
                                usedW1.add(-maxInd);
                            else usedW1.add(maxInd);
                          statsFin = evalExpansion(usedW1, usedW2,  supportsT1, supportsT2 , supportIntersection, numEntities);              
                          tmpRed = createRed(usedW1, usedW2, first, second, statsFin );  
                            
                        }
                        
                        //add clause to q2 if it improves JS with pv<maxP
                        
                        negated = 0;
                        maxInd = -1;
                        JSDiff = Double.NEGATIVE_INFINITY;
                        
                         for(int w2=j+1;w2<supportsT2.size();w2++){
                            
                            if(usedW2.contains(w2+1) || usedW2.contains(-(w2+1)))
                                continue;
                            
                            //try the positive and negated clause
                             usedW2.add(w2+1);
                             ArrayList<Double> stats = evalExpansion(usedW1, usedW2,  supportsT1, supportsT2 , supportIntersection, numEntities);           
                            
                              if(stats.get(4)>tmpRed.Jaccard && stats.get(2)<=this.maxSupport){
                                if(tmpRed.pval<this.maxPval){
                                    if(stats.get(5)<this.maxPval && (stats.get(4)-tmpRed.Jaccard)>JSDiff){
                                        maxInd = w2+1;
                                        JSDiff = stats.get(4)-tmpRed.Jaccard;
                                    }
                                }
                                else{
                                    if((stats.get(4)-tmpRed.Jaccard)>JSDiff && stats.get(2)<=this.maxSupport){
                                            maxInd = w2+1;
                                            JSDiff = stats.get(4)-tmpRed.Jaccard;
                                    }
                                }
                            }
                            else{
                                if(tmpRed.pval>this.maxPval && stats.get(2)<=this.maxSupport){
                                    if(stats.get(5)<this.maxPval){
                                         maxInd = w2+1;
                                         JSDiff = stats.get(4)-tmpRed.Jaccard;
                                    }
                                }
                            }
                              
                              usedW2.remove(w2+1);
                              
                              //negated
                              
                               usedW2.add(-(w2+1));
                             stats = evalExpansion(usedW1, usedW2,  supportsT1, supportsT2 , supportIntersection, numEntities);           
                            
                              if(stats.get(4)>tmpRed.Jaccard && stats.get(2)<=this.maxSupport){
                                if(tmpRed.pval<this.maxPval){
                                    if(stats.get(5)<this.maxPval && (stats.get(4)-tmpRed.Jaccard)>JSDiff){
                                        maxInd = w2+1;
                                        JSDiff = stats.get(4)-tmpRed.Jaccard;
                                        negated = 1;
                                    }
                                }
                                else{
                                    if((stats.get(4)-tmpRed.Jaccard)>JSDiff){
                                            maxInd = w2+1;
                                            JSDiff = stats.get(4)-tmpRed.Jaccard;
                                            negated = 1;
                                    }
                                }
                            }
                            else{
                                if(tmpRed.pval>this.maxPval && stats.get(2)<=this.maxSupport){
                                    if(stats.get(5)<this.maxPval && stats.get(3)<this.maxSupport){
                                         maxInd = w2+1;
                                         JSDiff = stats.get(4)-tmpRed.Jaccard;
                                         negated = 1;
                                    }
                                }
                            }
                              
                              usedW2.remove(-(w2+1));  
                        }
                        
                        //create final expansion
                        if(maxInd>-1){
                            if(negated == 1)
                                usedW2.add(-maxInd);
                            else usedW2.add(maxInd);
                          statsFin = evalExpansion(usedW1, usedW2,  supportsT1, supportsT2 , supportIntersection, numEntities);              
                          tmpRed = createRed(usedW1, usedW2, first, second, statsFin );  
                            
                        }
                        
                        //take the expansion with the maximum gain
                    }
                    
                    if(tmpRed.Jaccard>=this.minimalJaccard && tmpRed.pval<=this.maxPval && tmpRed.support>=this.minSupport && tmpRed.support<=this.maxSupport && tmpRed.querySupports.get(0)<this.maxSupport && tmpRed.querySupports.get(1)<this.maxSupport){
                          //tmpRed.normalizeRedescription();
                        tmpRed.queryStrings.add(tmpRed.createQueryString(0));
                        tmpRed.queryStrings.add(tmpRed.createQueryString(1));
                        tmpRed.queryStringsReReMi.add(tmpRed.createQueryReReMiCodesString(0));
                        tmpRed.queryStringsReReMi.add(tmpRed.createQueryReReMiCodesString(1));
                        
                        if(tmpRed.queryStrings.get(0).trim().equals("") || tmpRed.queryStrings.get(1).trim().equals(""))
                            continue;
                        
                        int tautology = 0;
                        
                        HashMap<String, HashSet<String>> catAttrVal = new HashMap<>();
                        
                        String q1 = tmpRed.queryStrings.get(0);
                            String q2 = tmpRed.queryStrings.get(1);
                            
                            String tmpQ1[] = q1.split(" OR ");
                            String tmpQ2[] = q2.split(" OR ");
                            
                            ArrayList<ArrayList<String>> qParts = new ArrayList<>();
                            
                            for(int z = 0; z<tmpQ1.length;z++){
                                if(tmpQ1[z].contains("NOT ")){
                                    String t = tmpQ1[z].replaceAll("NOT ", "");
                                    t = t.replaceAll("\\)","");
                                    t = t.replaceAll("\\(", "");
                                    ArrayList<String> part = new ArrayList<String>();
                                    
                                    String tt[] = t.trim().split(" AND ");
                                    
                                    for(int z1 = 0; z1<tt.length;z1++)
                                        part.add(tt[z1].trim());
                                   
                                     qParts.add(part);
                                }
                                else{
                                    if(!tmpQ1[z].trim().contains(" AND ") && !tmpQ1[z].trim().contains(">") && !tmpQ1[z].trim().contains("<=")){
                                         String attr = tmpQ1[z].trim().split("=")[0].trim();
                                         if(data.attribute(attr).isNominal()){
                                         String catVal = "";
                                          if(!catAttrVal.containsKey(attr))
                                              catAttrVal.put(attr, new HashSet<String>());
                                           catVal = tmpQ1[z].trim().split("=")[1].trim();
                                                   catAttrVal.get(attr).add(catVal);
                                         }
                                    }
                                             
                                }
                            }
                            
                            
                            
                            for(int z=0; z<qParts.size()-1;z++){//eliminate for Boolean as well
                               ArrayList<String> p1 = qParts.get(z);
                               for(int z2 = 0; z2<p1.size();z2++){
                                  // if(!p1.get(z2).contains("<") && !p1.get(z2).contains(">"))
                                    //   continue;
                                   String attr = null;
                                   double val = 0.0;
                                   String catVal = "";
                                   int lg = -1;
                                    if(p1.get(z2).contains("<=")){
                                           attr = p1.get(z2).split("<=")[0].trim();
                                           val = Double.parseDouble(p1.get(z2).split("<=")[1].trim());
                                           lg = 0;
                                                   }
                                    else if(p1.get(z2).contains(">")){
                                           attr = p1.get(z2).split(">")[0].trim();
                                           val = Double.parseDouble(p1.get(z2).split(">")[1].trim());
                                           lg = 1;
                                    }
                                    else if(p1.get(z2).contains("=")){
                                          attr = p1.get(z2).split("=")[0].trim();
                                          if(!catAttrVal.containsKey(attr))
                                              catAttrVal.put(attr, new HashSet<String>());
                                           catVal = p1.get(z2).split("=")[1].trim();
                                           for(int zz=0;zz<data.attribute(attr).numValues();zz++){
                                               if(!(data.attribute(attr).value(zz).trim().equals(catVal)))
                                                   catAttrVal.get(attr).add(data.attribute(attr).value(zz).trim());
                                           }
                                    }
                                           
                                   for(int z1=z+1;z1<qParts.size();z1++){
                                     ArrayList<String> p2 = qParts.get(z1);
                                        for(int z3 = 0; z3<p2.size();z3++){
                                           // if(!p2.get(z3).contains("<") && !p2.get(z3).contains(">"))
                                             //       continue;
                                            
                                   String attr1 = null;
                                   double val1 = 0.0;
                                   int lg1 = -1;
                                    if(p2.get(z3).contains("<=")){
                                           attr1 = p2.get(z3).split("<=")[0].trim();
                                           val1 = Double.parseDouble(p2.get(z3).split("<=")[1].trim());
                                           lg1 = 0;
                                                   }
                                    else if(p2.get(z3).contains(">")){
                                           attr1 = p2.get(z3).split(">")[0].trim();
                                           val1 = Double.parseDouble(p2.get(z3).split(">")[1].trim());
                                           lg1 = 1;
                                    }
                                     /*else if(p1.get(z2).contains("=")){
                                          attr = p1.get(z2).split("=")[0].trim();
                                          if(!catAttrVal.containsKey(attr))
                                              catAttrVal.put(attr, new HashSet<String>());
                                           catVal = p1.get(z2).split("=")[1].trim();
                                           for(int zz=0;zz<data.attribute(attr).numValues();zz++){
                                               if(!(data.attribute(attr).value(zz).trim().equals(catVal)))
                                                   catAttrVal.get(attr).add(data.attribute(attr).value(zz).trim());
                                           }
                                    }*/
                                    
                                            if(attr.equals(attr1) && val == val1 && lg == (1-lg1)){
                                                tautology = 1;
                                                break;
                                            }
                                                
                                        }
                                          if(tautology == 1)
                                              break;
                                  }
                                   if(tautology == 1)
                                              break;
                            }
                               if(tautology == 1)
                                       break;
                           }
                            
                           
                          if(tautology == 1){
                              System.out.println("Tautology detected!");
                              System.out.println(tmpRed.queryStrings.get(0));
                              continue;
                          }
                          
                           //categorical check
                            Iterator<String> it = catAttrVal.keySet().iterator();
                            
                            while(it.hasNext()){
                                int taut = 1;
                                String at = it.next();
                                HashSet<String> vals = catAttrVal.get(at);
                                for(int zz=0;zz<data.attribute(at).numValues();zz++)
                                    if(!vals.contains(data.attribute(at).value(zz))){
                                        taut = 0;
                                        break;
                                    }
                                
                                if(taut == 1){
                                    tautology = 1;
                                    break;
                                }
                            }
                            
                             if(tautology == 1){
                              System.out.println("Tautology detected!");
                              System.out.println(tmpRed.queryStrings.get(0));
                              continue;
                          }
                             
                           catAttrVal.clear();
                        
                          
                         qParts = new ArrayList<>();
                            
                            for(int z = 0; z<tmpQ2.length;z++){
                                if(tmpQ2[z].contains("NOT ")){
                                    String t = tmpQ2[z].replaceAll("NOT ", "");
                                    t = t.replaceAll("\\)","");
                                    t = t.replaceAll("\\(", "");
                                    ArrayList<String> part = new ArrayList<String>();
                                    
                                    String tt[] = t.trim().split(" AND ");
                                    
                                    for(int z1 = 0; z1<tt.length;z1++)
                                        part.add(tt[z1].trim());
                                   
                                     qParts.add(part);
                                }
                                 else{
                                    if(!tmpQ2[z].trim().contains(" AND ") && !tmpQ2[z].contains(">") && !tmpQ2[z].contains("<=")){
                                         String attr = tmpQ2[z].trim().split("=")[0].trim();
                                         if(data1.attribute(attr).isNominal()){
                                         String catVal = "";
                                          if(!catAttrVal.containsKey(attr))
                                              catAttrVal.put(attr, new HashSet<String>());
                                           catVal = tmpQ2[z].trim().split("=")[1].trim();
                                            catAttrVal.get(attr).add(catVal);
                                         }
                                    }
                                             
                                }
                            }
                            
                            for(int z=0; z<qParts.size()-1;z++){
                               ArrayList<String> p1 = qParts.get(z);
                               for(int z2 = 0; z2<p1.size();z2++){
                                  // if(!p1.get(z2).contains("<") && !p1.get(z2).contains(">"))
                                    //   continue;
                                   String attr = null;
                                   double val = 0.0;
                                   int lg = -1;
                                    if(p1.get(z2).contains("<=")){
                                           attr = p1.get(z2).split("<=")[0].trim();
                                           val = Double.parseDouble(p1.get(z2).split("<=")[1].trim());
                                           lg = 0;
                                                   }
                                    else if(p1.get(z2).contains(">")){
                                           attr = p1.get(z2).split(">")[0].trim();
                                           val = Double.parseDouble(p1.get(z2).split(">")[1].trim());
                                           lg = 1;
                                    }
                                    else if(p1.get(z2).contains("=")){
                                          attr = p1.get(z2).split("=")[0].trim();
                                          if(!catAttrVal.containsKey(attr))
                                              catAttrVal.put(attr, new HashSet<String>());
                                           String catVal = p1.get(z2).split("=")[1].trim();
                                           for(int zz=0;zz<data1.attribute(attr).numValues();zz++){
                                               if(!(data1.attribute(attr).value(zz).trim().equals(catVal)))
                                                   catAttrVal.get(attr).add(data1.attribute(attr).value(zz).trim());
                                           }
                                    }
                                           
                                   for(int z1=z+1;z1<qParts.size();z1++){
                                     ArrayList<String> p2 = qParts.get(z1);
                                        for(int z3 = 0; z3<p2.size();z3++){
                                            if(!p2.get(z3).contains("<") && !p2.get(z3).contains(">"))
                                                    continue;
                                            
                                   String attr1 = null;
                                   double val1 = 0.0;
                                   int lg1 = -1;
                                    if(p2.get(z3).contains("<=")){
                                           attr1 = p2.get(z3).split("<=")[0].trim();
                                           val1 = Double.parseDouble(p2.get(z3).split("<=")[1].trim());
                                           lg1 = 0;
                                                   }
                                    else if(p2.get(z3).contains(">")){
                                           attr1 = p2.get(z3).split(">")[0].trim();
                                           val1 = Double.parseDouble(p2.get(z3).split(">")[1].trim());
                                           lg1 = 1;
                                    }
                                    
                                            if(attr.equals(attr1) && val == val1 && lg == (1-lg1) ){
                                                tautology = 1;
                                                break;
                                            }
                                                
                                        }
                                          if(tautology == 1)
                                              break;
                                  }
                                   if(tautology == 1)
                                              break;
                            }
                               if(tautology == 1)
                                       break;
                           }
                            
                             if(tautology == 1){
                                 System.out.println("Tautology detected!");
                                 System.out.println(tmpRed.queryStrings.get(0));                         
                              continue;
                             }
                             
                             //check tautology for categorical
                          //categorical check
                            it = catAttrVal.keySet().iterator();
                            
                            while(it.hasNext()){
                                int taut = 1;
                                String at = it.next();
                                HashSet<String> vals = catAttrVal.get(at);
                                for(int zz=0;zz<data1.attribute(at).numValues();zz++)
                                    if(!vals.contains(data1.attribute(at).value(zz))){
                                        taut = 0;
                                        break;
                                    }
                                
                                if(taut == 1){
                                    tautology = 1;
                                    break;
                                }
                            }
                            
                             if(tautology == 1){
                              System.out.println("Tautology detected!");
                              System.out.println(tmpRed.queryStrings.get(0));
                              continue;
                          }
                             
                           catAttrVal.clear();

                        int duplicate = 0;
                        //deal with the case NOT (At1>c AND At2) OR NOT(AT1<=c AND At3) //tautology
                        for(int zc = 0;zc<result.size();zc++){
                            if(result.get(zc).queryStrings.get(0).trim().equals(tmpRed.queryStrings.get(0).trim()) && (result.get(zc).queryStrings.get(1).trim().equals(tmpRed.queryStrings.get(1).trim()))){
                                                    duplicate = 1;
                                                     break;
                                 }
                            
                            if(result.get(zc).Jaccard == tmpRed.Jaccard &&  result.get(zc).support == tmpRed.support && result.get(zc).supportUnion == tmpRed.supportUnion && result.get(zc).pval == tmpRed.pval && result.get(zc).querySupports.get(0) == tmpRed.querySupports.get(0) && result.get(zc).querySupports.get(1) == tmpRed.querySupports.get(1) ){
                                duplicate = 1;
                                break;
                            }
                            
                        }
                        
                        if(duplicate == 0){
                            
                            if(flip == 1){
                                 Collections.reverse(tmpRed.queryStrings);
                                 Collections.reverse(tmpRed.queryStringsReReMi);
                                 Collections.reverse(tmpRed.querySupports);
                                 Collections.reverse(tmpRed.redNodes);
                            }
                            
                             result.add(tmpRed);
                        }
                    }
                  
                    usedW1.clear();
                    usedW2.clear();
              }
            }
            
         
         return result;
     }
     
    //use the trick instead of Exp mechanism for numerical stability
         
    public String toString(int treeNum) {                    

        if(treeNum == 0)
            return toString(RootFirst);
        else return toString(RootSecond);
       
    }

	protected String toString(Node node) {
		
		int level = node.depth;

		StringBuffer text = new StringBuffer();

		if (node.isLeaf == true) {                              

			text.append("  [" + node.data.numInstances() + "]");
			text.append(": ").append(
					node.data.classAttribute().value((int) Utils.maxIndex(node.dist)));
			
			text.append("   Counts  " + distributionToString(node.dist));
			
		} else {                                                
			
			text.append("  [" + node.data.numInstances() + "]");
			for (int j = 0; j < node.children.length; j++) {
				
				text.append("\n");
				for (int i = 0; i < level; i++) {
					text.append("|  ");
				}
				
                               if(!node.splitAttr.isNumeric()){ 
				text.append(node.splitAttr.name())
					.append(" = ")
					.append(node.splitAttr.value(j));
                               }
                               else{
                                   if(j == 0)
                                   text.append(node.splitAttr.name())
					.append(" <= ")
                                        .append(node.splitValue);
                                   else{
                                       text.append(node.splitAttr.name())
					.append(" > ")
                                        .append(node.splitValue);
                                   }
                               }
				
				text.append(toString(node.children[j]));
			}
		}
		return text.toString();
	}

    private String distributionToString(double[] distribution)
    {
           StringBuffer text = new StringBuffer();
           text.append("[");
           for (double d:distribution)
                  text.append(String.format("%.2f", d) + "; ");
           text.append("]");
           return text.toString();             
    }
    
}
