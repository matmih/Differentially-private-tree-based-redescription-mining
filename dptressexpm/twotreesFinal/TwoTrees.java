/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package dptressexpm.twotreesFinal;

import java.io.Serializable;
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
 * @author Matej Mihelcic, University of Eastern Finland implemented the dptressxpm.twotreesFinal package as an extension of the 
 * original tree package decTreeWholeDPGeneral created for the manuscript: "Embedding differential privacy in decision tree
algorithm with different depths" by authors Xuanyu BAI , Jianguo YAO*, Mingxuan YUAN, Ke DENG, Xike Xie and Haibing Guan.
 */

//toDo
//create a function that takes the first tree a shallow copy of the data (instances)
//deletes the old class (first save the class index, then set to -1, then delete attribute at index
//then add new attribute, then add values of the attribute, recursive call using
//sets of instance indices
//call tree construction with the new dataset (first time randomTree), other occasions just
//updateTree function
//if first tree is modified update the second tree
//if second tree is modified do nothing 
//copmute new paird tree gini score

//add numeric attributes and missing data into this package. Fully general DP RM algo

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
	private Set<Node> InnerNodesFirst = new HashSet<Node>();
        private Set<Node> InnerNodesSecond = new HashSet<Node>();

	GiniIndexDirectAdd gini = new GiniIndexDirectAdd();
	
	public class Node implements Serializable{
		
		private static final long serialVersionUID = 1L;     

		public Instances data;
		
		public int depth, complement = 0;
		
		public Attribute splitAttr;
                public double splitValue;
		
		public Node[] children;
		
		public Node parent;
		
		public int index;
                public int li;
		
		public boolean isLeaf;
		
		public double[] count;            
		public double[] dist;             
		
		public void updateDist(){                         
			dist = count.clone();                          
			if( Utils.sum(dist) != 0.0)                    
				Utils.normalize(dist);         
		}
		
		public Node(Instances data, Node parent, int index){
			
			this.data = data;
			this.parent = parent;
			this.index = index;

			if(parent!=null)
				this.depth = parent.depth + 1;
			else
				this.depth = 1;
			
			double[] tempcount = new double[data.numClasses()];                
			Enumeration<Instance> instEnum = data.enumerateInstances();  
			while(instEnum.hasMoreElements()){                             
				Instance inst = (Instance)instEnum.nextElement();            
				tempcount[(int)inst.classValue()]++;                      
			}
				
			this.count = tempcount;                                   
			this.dist = count.clone();                      
			if(Utils.sum(this.dist) != 0.0)                       
				Utils.normalize(this.dist);                   
		}

		public Node(Node another){                              
			
			this.data = new Instances(another.data); 
			this.depth = another.depth;                         
			this.isLeaf = another.isLeaf;
			this.count = another.count;
			this.dist = another.dist;
			this.splitAttr = another.splitAttr;
                        this.splitValue = another.splitValue;
        }
                
                public Node(){
                    this.depth = 0;                         
			this.isLeaf = true;
			this.count = null;
			this.dist = null;
			this.splitAttr = null;
                        this.parent = null;
                        this.complement = 0;
                }
                
                public void copyNodeFull(Node another){
                    this.data = new Instances(another.data); 
			this.depth = another.depth;                         
			this.isLeaf = another.isLeaf;
			this.count = another.count;
			this.dist = another.dist;
			this.splitAttr = another.splitAttr;
                        this.parent = another.parent;
                        this.index = another.index;
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
    				
    				text.append(node.splitAttr.name())
    					.append(" = ")
    					.append(node.splitAttr.value(j));
    				
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
        
        //sample size, fraction of the input data
        public HashSet<Pair<Attribute, Double>> createAttrList(Instances data, double sampleSize){
            HashSet<Pair<Attribute, Double>> tmp = new HashSet<>();
            Random rand = new Random();
            
            if(sampleSize<0 || sampleSize>1)
                sampleSize = 0.2;
            
            int numInst = (int)(sampleSize * data.numInstances());
            HashSet<Integer> sampled = new HashSet<>();
            
            Instances tmpInstances = new Instances(data,0);
            
            
            for(int i=0;i<numInst;i++){
                
                if(sampled.size() == data.numInstances())
                    break;
                
                int index = rand.nextInt(data.numInstances());
                
                if(!sampled.contains(index)){
                    sampled.add(index);
                    tmpInstances.add(data.get(index));
                }
                else{
                    while(sampled.contains(index)){
                        index = rand.nextInt(data.numInstances());
                    }
                    
                     sampled.add(index);
                    tmpInstances.add(data.get(index));
                }
            }
            
            //create tmp from tmpInstances
            for(int i=0;i<tmpInstances.numAttributes();i++){
                HashSet<Double> values = new HashSet<>();
                
                 Attribute a = tmpInstances.attribute(i);
                
                 if(a.isNumeric()){
                    for(int j=0;j<tmpInstances.numInstances();j++){
                        if(!tmpInstances.get(j).isMissing(a))
                            values.add(tmpInstances.get(j).value(i));
                    }

                for(double d:values){
                    tmp.add(new Pair<>(a,d));
                }
              }
                 else{
                     tmp.add(new Pair<>(a,Double.NEGATIVE_INFINITY));
                 }
                
            }
            
            return tmp;
        }
        
        public void buildClassifierTest(Instances data) throws Exception {
            	HashSet<Pair<Attribute,Double>> allAttributesGlobal = this.createAttrList(data, 0.2);
                HashSet<Pair<Attribute,Double>> allAttributes = new HashSet<>();
       
        //create the code loading all attribute-value combinations from the data
        
      for(Pair<Attribute,Double> p:allAttributesGlobal)
                allAttributes.add(p);
        
        Attributes = new HashSet<Pair<Attribute,Double>>(allAttributes);
        Instances dataCopy = new Instances(data);
        Attribute a;
        
        RootFirst = new Node(data, null, 0); 
        Set<Pair<Attribute,Double>> Attrs = new HashSet<Pair<Attribute,Double>>(Attributes);
        
            //modify splitting function to create a split
                 randomSplitTree(RootFirst, Attrs,0);  
        }

	public void createTrees(Instances data, Instances data1) throws Exception {
                HashSet<Pair<Attribute,Double>> allAttributesGlobal = this.createAttrList(this.view1Data, 0.8);
                HashSet<Pair<Attribute,Double>> allAttributes = new HashSet<>();
		
		 for(Pair<Attribute,Double> p:allAttributesGlobal)
                      allAttributes.add(p);
   
        Attributes = new HashSet<Pair<Attribute,Double>>(allAttributes);
        Instances dataCopy = null, data2Copy = new Instances(data1);
        Attribute a;  
        double attributeValue;
        
         HashSet<Pair<Attribute,Double>> allAttributesGlobal1 = this.createAttrList(this.view2Data, 0.8);
                HashSet<Pair<Attribute,Double>> allAttributes1 = new HashSet<>();
		
		 for(Pair<Attribute,Double> p:allAttributesGlobal1)
                      allAttributes1.add(p);
        
        Attributes1 = new HashSet<Pair<Attribute, Double>>(allAttributes1);
        
        RootFirst = new Node(data, null, 0);   
        RootSecond = null; //new Node(data, null, 0); 
        
        splitbudget = Epsilon.divide(BigDecimal.valueOf(2),MATH_CONTEXT);//add 3 if MCMC used for redescription creation
        noisebudget = splitbudget;
        splitmarkovbudget = Epsilon.subtract(noisebudget,MATH_CONTEXT);
        
        Set<Pair<Attribute,Double>> Attrs = new HashSet<Pair<Attribute,Double>>(Attributes);
        Set<Pair<Attribute, Double>> Attrs1 = new HashSet<Pair<Attribute,Double>>(Attributes1);
        randomSplitTree(RootFirst, Attrs,0);
        //System.out.println(RootFirst.toString(RootFirst));
        
        Attrs = new HashSet<Pair<Attribute,Double>>(Attributes);
        
        dataCopy = addTargets(data);
        data2Copy = copyTargets(dataCopy,data2Copy);
        //add the same targets to the second dataset, learn tree2 on the second dataset
        
        RootSecond = new Node(data2Copy, null, 0); 
        randomSplitTree(RootSecond,Attrs1,1);
        
        numEntities = RootFirst.data.numInstances();
        
        boolean equilibrium = false;
        int iteration = 0;
        double scores[] = new double[2];
        double explogprobabilities[] = null;
        Random unif = new Random();
    
        while(iteration < MaxIteration && !equilibrium){
        	double initialscore = 0;
        	double laterscore = 0;
                double finscore = 0;
                
                //randomly choose a node from any tree to change
        	
        	initialscore = gini.score(RootFirst,RootSecond);//implement gini score for pairs of trees
        	
                Object nodesFirst[] = InnerNodesFirst.toArray();
                Object nodesSecond[] = InnerNodesSecond.toArray();
                
                int ts = nodesFirst.length + nodesSecond.length;
                
                if(ts == 0)
                    return;
                //watch the case where there are no inner nodes, just return
                int index = Random.nextInt(ts);
                int treeChoise = -1;
                
                Node oldnode = null;
                
                if(index<nodesFirst.length){
                    treeChoise = 0;
                    oldnode = (Node)nodesFirst[index];
                }
                else{ 
                    treeChoise = 1;
                    oldnode = (Node)nodesSecond[index - nodesFirst.length];
                }
               
        	//Node oldnode = (Node)InnerNodesFirst.toArray()[Random.nextInt(InnerNodesFirst.size())]; //change   

        	Node newnode = new Node(oldnode);
                           
        	Set<Pair<Attribute,Double>> subattrs = null;
                        
                if(treeChoise == 0)
                        subattrs = new HashSet<Pair<Attribute,Double>>(Attributes);  
                else subattrs = new HashSet<Pair<Attribute,Double>>(Attributes1);  
                
        	deleteParAttr(oldnode,subattrs);
        	deleteChiAttr(oldnode,subattrs);   
                
        	if(subattrs.size()==0){
        		iteration++;
        		continue;
        	}
        	else{
                	 Pair<Attribute,Double> p = (Pair<Attribute,Double>)subattrs.toArray()[Random.nextInt(subattrs.size())];
                         a = p.getFirst();
                         attributeValue = p.getSecond();
        	}
        
        	Instances[] Parts = partitionByAttr(newnode.data,a, attributeValue);     
	     	Node[] Children = null;
                        
                        if(a.isNumeric())
                            Children = new Node[2];
                        else
                            Children = new Node[a.numValues()];    
	     	
                    newnode.splitAttr = a;
                    newnode.splitValue = attributeValue;
		    newnode.children = Children;
			
			for(int k=0; k < Parts.length; k++){
				Children[k] = new Node(Parts[k], newnode, k);
				
				makeLeafNode(Children[k]);
			}
		
			int minlen = (oldnode.children.length<newnode.children.length) ? oldnode.children.length : newnode.children.length;
			
			if(minlen==oldnode.children.length){          
				for(int l=0; l<minlen;l++){
					replaceNode(oldnode.children[l],newnode.children[l]);
			    }
				if(minlen!=newnode.children.length){
					for(int l=minlen; l<newnode.children.length; l++){
						Set<Pair<Attribute,Double>> subtreeAttrs = null;
                                                if(treeChoise == 0)
                                                        subtreeAttrs = new HashSet<Pair<Attribute,Double>>(Attributes);
                                                else subtreeAttrs = new HashSet<Pair<Attribute,Double>>(Attributes1);
						randomSplitSubtree(newnode.children[l], subtreeAttrs, treeChoise);
					}
				}
			}
			else if(minlen==newnode.children.length){
				for(int l=0; l<minlen;l++){
					replaceNode(oldnode.children[l],newnode.children[l]);
			    }
			}
			
			if(oldnode.parent==null){
				newnode.parent = null;
				newnode.index = 0;
                                if(treeChoise == 0){
                                    RootFirst = newnode;//change
                                }
                                else{
                                    RootSecond = newnode;
                                }
			}
			else{
			    Node oldnodepar = oldnode.parent;
				newnode.parent = oldnodepar;
				newnode.index = oldnode.index;
				oldnodepar.children[oldnode.index] = newnode;
			}
		
                        //if first tree is changed a necesary temporary change in targets of the 
                        //second tree needs to be made to compute the impact on the score...
                        
                        if(treeChoise==0){
                            renumerateLeaves(RootFirst);
                                dataCopy = addTargets(data);
                                data2Copy = copyTargets(dataCopy,data2Copy);
                                updateTargetDistribution(RootSecond, data2Copy);
                               }
                        
			laterscore = gini.score(RootFirst, RootSecond);//change
			
                        //compute using the trick, more numerically stable
			//double initialpro = expProbability(initialscore,splitmarkovbudget);
			//double laterpro = expProbability(laterscore,splitmarkovbudget);
                         
                        scores[0] = initialscore; scores[1] = laterscore;
                        
                        explogprobabilities = expLogProbability(scores, splitmarkovbudget);  
                        
                         double gs[] = new double[2];
                
                            for(int ig=0;ig<gs.length;ig++){
                                double rn = unif.nextDouble();
                                gs[ig] = -Math.log(-Math.log(rn));
                               // if(gs[ig]<0)
                                 //   gs[ig] = 0.0;
                            }
			
                            if(gs[0] < 0 && gs[1] < 0){
                                double tmp = Math.abs(gs[0]);
                                gs[0] = Math.abs(gs[1]);
                                gs[1] = Math.abs(tmp);
                            }
                            else if(gs[0]<0 && gs[1]>=0){
                                gs[0] = 0;
                            }
                            else if(gs[1]<0 && gs[0]>=0){
                                gs[1] = 0;
                            }
                            
                             int flag = 0;
                            double max = Double.NEGATIVE_INFINITY;
        System.out.println("log scores: ");
        System.out.println(explogprobabilities[0]+" "+explogprobabilities[1]);
        
        double initialpro = explogprobabilities[0]+gs[0];
        double laterpro = explogprobabilities[1]+gs[1];
        
        System.out.println("Init prob: "+initialpro);
        System.out.println("Later prob: "+laterpro);
          
			double ratio = (double)(laterpro/initialpro); 
                        if(initialpro == 0)
                            ratio = 1.0;
                        //get info gain in [0,1] to be usable in here
                        
                        System.out.println("Tree choice: "+treeChoise);
                        System.out.println("Change: "+flag);
                        System.out.println("Ratio: "+ratio);
                        
			if(ratio>=1){
                               if(treeChoise==0){
				removeInnerNodes(oldnode,0);//change
				addInnerNodes(newnode,0);//change
                                renumerateLeaves(RootFirst);
                                dataCopy = addTargets(data);
                                data2Copy = copyTargets(dataCopy,data2Copy);
                                updateTargetDistribution(RootSecond, data2Copy);
                               }
                               else{
                                   removeInnerNodes(oldnode,1);//change
				addInnerNodes(newnode,1);//change
                               }

				double totalscore = gini.score(RootFirst,RootSecond); //change        
				equilibrium = isEquilibrium(totalscore);     
                                System.out.println("total score: "+totalscore);
                                finscore = totalscore;
			}
			else{
				boolean replace = false;
				double randomdouble = Random.nextDouble();
                                
				if(randomdouble<ratio){
					replace = true;
				}
                                
                                System.out.println("replace: "+replace);
                                
				if(replace){

                                    if(treeChoise==0){
					removeInnerNodes(oldnode,0);//change
					addInnerNodes(newnode,0);//change
                                        renumerateLeaves(RootFirst);
                                        dataCopy = addTargets(data);
                                        data2Copy = copyTargets(dataCopy,data2Copy);
                                        updateTargetDistribution(RootSecond, data2Copy);
                                    }
                                    else{
                                        removeInnerNodes(oldnode,1);//change
					addInnerNodes(newnode,1);//change
                                    }

					double totalscore = gini.score(RootFirst,RootSecond);//change          
					equilibrium = isEquilibrium(totalscore);     
                                        finscore = totalscore;
				}
				else{

					if(oldnode.parent==null){
                                            if(treeChoise == 0){
						RootFirst = oldnode;//change
                                            }
                                            else{
                                                RootSecond = oldnode;
                                            }
					}
					else{
						Node newnodepar = newnode.parent;
						oldnode.parent = newnodepar;
						oldnode.index = newnode.index;
						newnodepar.children[newnode.index] = oldnode;
					}
                                        
                                        if(treeChoise==0){
                                            renumerateLeaves(RootFirst);
                                            dataCopy = addTargets(data);
                                            data2Copy = copyTargets(dataCopy,data2Copy);
                                            updateTargetDistribution(RootSecond, data2Copy);
                                         }
                                        
                                        finscore = gini.score(RootFirst,RootSecond);
				}
			}
                        
                 System.out.println("iteration: "+iteration);
                 System.out.println("initial score: "+initialscore);
                 System.out.println("later score: "+laterscore);
                 System.out.println("final score: "+finscore);
                 System.out.println();

        	iteration++;                                 
        }

        //instead of addNoise, createRedescriptons required
        //addNoise(RootFirst, noisebudget); //change                                
        
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

	private void randomSplitTree(Node node, Set<Pair<Attribute,Double>> Attrs, int numTree){

                   if(numTree == 0)
                        Attrs = new HashSet<Pair<Attribute,Double>>(Attributes);
            else Attrs = new HashSet<Pair<Attribute,Double>>(Attributes1);

		if(node.depth >= MaxDepth){                     
			makeLeafNode(node);
			return;
		}
		if(node.data.numInstances()<=minNodeSize){
			makeLeafNode(node);
			return;
		}
		for(int i=0; i< node.count.length; i++){        
			if(node.count[i] == node.data.numInstances()){
				makeLeafNode(node);
				return;
			}
		}
		
		deleteParAttr(node,Attrs);
		if(Attrs.size()==0){
			makeLeafNode(node);
			return;
		}
	
		makeInnerNode(node);
	
                if(numTree == 0)
                    InnerNodesFirst.add(node);   //change
                else 
                    InnerNodesSecond.add(node);

                Pair<Attribute,Double> SAInfo = (Pair<Attribute,Double>)Attrs.toArray()[Random.nextInt(Attrs.size())];
		Attribute splitAttr = SAInfo.getKey();//(Attribute)Attrs.toArray()[Random.nextInt(Attrs.size())];  
		double splitValue = SAInfo.getValue();
                
		Instances[] parts = partitionByAttr(node.data, splitAttr, splitValue);           
		Node[] children = null;
                
                        if(splitAttr.isNumeric())
                            children = new Node[2];
                        else
                            children = new Node[splitAttr.numValues()];        
		
		node.splitAttr = splitAttr; 
                node.splitValue = splitValue;
		node.children = children;
		
		for(int i=0; i < parts.length; i++){
			children[i] = new Node(parts[i], node, i);  
			randomSplitTree(children[i], Attrs, numTree);     
		}		
	}
	
	private void makeLeafNode(Node node){
		
		node.splitAttr = null;   
                node.splitValue = Double.NEGATIVE_INFINITY;
		node.children  = null;                                            
		node.isLeaf = true;	
                node.li = this.numLeafs;
                this.numLeafs++;
	}
	
	private void makeInnerNode(Node node){
                node.li = -1;
		node.isLeaf = false;	
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
	
	private void deleteParAttr(Node node, Set<Pair<Attribute,Double>> Attrs){
		
		while(node.parent != null){
                    Pair<Attribute, Double> a = new Pair(node.parent.splitAttr,node.parent.splitValue);
			Attrs.remove(a);
			node = node.parent;
		}
		return;
	}

	private void deleteChiAttr(Node node, Set<Pair<Attribute,Double>> Attrs){
	
		if(node.isLeaf==true)
			return;
		Attrs.remove(new Pair(node.splitAttr,node.splitValue));

		for(Node child: node.children){
				
			deleteChiAttr(child,Attrs);		
		}
	}

	private void replaceNode(Node oldnode, Node newnode){
		
		if(oldnode.isLeaf==true){
			makeLeafNode(newnode);
                        newnode.li = oldnode.li;
			return;
		}
		
		makeInnerNode(newnode);
		
		Attribute splitAttr = oldnode.splitAttr;
                double splitValue = oldnode.splitValue;
		
		newnode.splitAttr = splitAttr;
                newnode.splitValue = oldnode.splitValue;
		newnode.index = oldnode.index;
		
		Node[] children = null;
                
                    if(splitAttr.isNumeric())
                        children = new Node[2];
                    else children = new Node[splitAttr.numValues()];
                    
		Instances[] parts = partitionByAttr(newnode.data, splitAttr, splitValue);
		
		newnode.children = children;
	
		for(int i=0;i<parts.length;i++){
			children[i] = new Node(parts[i],newnode,i);
			
			replaceNode(oldnode.children[i],children[i]);
		}
	}
	
	private void removeInnerNodes(Node node, int numTree){
		
		if(node.isLeaf==true)
			return;
		for(Node child: node.children){
			removeInnerNodes(child, numTree);
		}
                if(numTree == 0)
                    InnerNodesFirst.remove(node);
                else InnerNodesSecond.remove(node);
	}

	private void addInnerNodes(Node node, int numTree){
		
		if(node.isLeaf==true)
			return;
		for(Node child: node.children){
			addInnerNodes(child, numTree);
		}
                if(numTree == 0)
                    InnerNodesFirst.add(node);
                else InnerNodesSecond.add(node);
	}

	private void randomSplitSubtree(Node node, Set<Pair<Attribute, Double>> Attrs, int treeChoice){
		
            if(treeChoice == 0)
		Attrs = new HashSet<Pair<Attribute,Double>>(Attributes);//something strange
            else Attrs = new HashSet<Pair<Attribute,Double>>(Attributes1);

		if(node.depth >= MaxDepth){                     
			makeLeafNode(node);
			return;
		}
		if(node.data.numInstances()<=minNodeSize){//add here min support condition
			makeLeafNode(node);
			return;
		}
		for(int i=0; i< node.count.length; i++){        
			if(node.count[i] == node.data.numInstances()){
				makeLeafNode(node);
				return;
			}
		}
		
		deleteParAttr(node,Attrs);
		if(Attrs.size()==0){
			makeLeafNode(node);
			return;
		}
	
		makeInnerNode(node);
		//change
                Pair<Attribute,Double> attrInfo = (Pair<Attribute,Double>)Attrs.toArray()[Random.nextInt(Attrs.size())];
		//Attribute splitAttr = (Attribute)Attrs.toArray()[Random.nextInt(Attrs.size())];  
		Attribute splitAttr = attrInfo.getFirst();
                double splitValue = attrInfo.getValue();
                        
		Instances[] parts = partitionByAttr(node.data, splitAttr, splitValue);           
		Node[] children = null;//new Node[splitAttr.numValues()];      
                
                if(splitAttr.isNumeric()){
                    children = new Node[2];
                }
                else children = new Node[splitAttr.numValues()]; 
		
		node.splitAttr = splitAttr;                       
		node.children = children;
                node.splitValue = splitValue;
		
		for(int i=0; i < parts.length; i++){
			children[i] = new Node(parts[i], node, i);  
			randomSplitSubtree(children[i], Attrs, treeChoice);     
		}
	}
	
    private void addNoise(Node node, BigDecimal budget){          

    	if(node.isLeaf == true)                                  
    	{ 
    		addNoiseDistribution(node.count, budget);   
    		node.updateDist();                                
	    	return;
    	}

    	for(Node child : node.children)                    
        {
        	addNoise(child, budget);
        }
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
    
        private double[] expLogProbability(double [] score, BigDecimal epsilon)              
  	{
                double res[] = new double[score.length];
                
                for(int i=0;i<res.length;i++)
                    res[i] = epsilon.doubleValue() * score[i] / (2 * rScoreSen);
  		//return Math.exp(epsilon.doubleValue() * score / (2 * GiniSen));
                return res;
  	}  
    

  	private double expProbability(double score, BigDecimal epsilon)        
  	{
  		return Math.exp(epsilon.doubleValue() * score / (2 * GiniSen));
  	}                  
  	
  	final int BufferSize = 500;   
  	double[] ScoreBuffer = new double[BufferSize];

  	int InitPointer = 0;           
  	int Pointer = 0;               
  	double variance = -1;                 
  	
  	private boolean isEquilibrium(double newScore)
  	{
            
            if(newScore>=1.6)
                return true;
            
  		if(InitPointer == BufferSize){
  			variance = Utils.variance(ScoreBuffer);
                         System.out.println("Variance: "+variance);
  			if(variance < EquilibriumThreshold){
  				return true;
  			}
  			InitPointer++;
  		}
  		if( InitPointer < BufferSize ){                 
  			ScoreBuffer[InitPointer++] = newScore;       
  			
  			return false;                                    
  		}
  		if(Pointer==BufferSize)
  			Pointer = 0;
  		ScoreBuffer[Pointer++] = newScore;                                 
  		
  		variance = Utils.variance(ScoreBuffer);     
                 System.out.println("Variance: "+variance);

  		if(variance < EquilibriumThreshold){         
  			InitPointer = 0;
  			Pointer = 0;                                
  			return true; 
  		}
  		return false;                                   
  	}
        
        private boolean isEquilibriumRed(double newScore, Redescription red)
  	{
            
            
            if(red.Jaccard >= this.minimalJaccard && red.pval<= this.maxPval && red.support>=this.minSupport && red.support<=this.maxSupport){
                return true;
            }
            
            //if(newScore==1.0)
              //  return true;
            
  		if(InitPointer == BufferSize){
  			variance = Utils.variance(ScoreBuffer);
                         System.out.println("Variance: "+variance);
  			if(variance < EquilibriumThreshold){
  				return true;
  			}
  			InitPointer++;
  		}
  		if( InitPointer < BufferSize ){                 
  			ScoreBuffer[InitPointer++] = newScore;       
  			
  			return false;                                    
  		}
  		if(Pointer==BufferSize)
  			Pointer = 0;
  		ScoreBuffer[Pointer++] = newScore;                                 
  		
  		variance = Utils.variance(ScoreBuffer);     
                 System.out.println("Variance: "+variance);

  		if(variance < EquilibriumThreshold){         
  			InitPointer = 0;
  			Pointer = 0;                                
  			return true; 
  		}
  		return false;                                   
  	}
  	
    public double classifyInstance(Instance instance, int numTree) throws NoSupportForMissingValuesException { 	
    	
    	assert( instance.hasMissingValue() == false);
    	
    	Node node = null;
        
        if(numTree == 0)
                node = RootFirst;
        else node = RootSecond;
    	while(node.isLeaf == false){                          
    		Attribute attr = node.splitAttr;
    		node = node.children[ (int)instance.value(attr) ];
    	}
    	
    	return Utils.maxIndex(node.dist);          
    }

	public double[] distributionForInstance(Instance instance, int treeNum) throws NoSupportForMissingValuesException {	
		
		assert( instance.hasMissingValue() == false);
		
		Node node = null;
                
                if(treeNum == 0)
                    node = RootFirst;
                else node = RootSecond;
    	while(node.isLeaf == false){                     
    		Attribute attr = node.splitAttr;
    		node = node.children[ (int)instance.value(attr) ];
    	}
    	
    	return node.dist;                           
	} 
        
        
        public int countLeafs(Node node, int count){
            
            if(node.isLeaf){
                count = count+1;
                return count;
            }
            
            for(int i=0;i<node.children.length;i++)
                count = countLeafs(node.children[i],count);
            
            return count;
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
            
             HashSet<Integer> toRemove = new HashSet<>();
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
            
             HashSet<Integer> toRemove = new HashSet<>();
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
    
    public Redescription createRedescription(Instances data, Instances data1){
        Redescription tmp = new Redescription(numClausesRed);
        
        boolean equilibrium = false;
        int iteration = 0;
        
        //list all leaf nodes from tree1 + empty
        //list all leaf nodes from tree2 + empty
        
         ArrayList<Node> first = getLeafs(0, RootFirst, new ArrayList<Node>());
        ArrayList<Node> second = getLeafs(1, RootSecond, new ArrayList<Node>());
        
        
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
        
        Random rand = new Random();
        int r1, qn;
        
        HashSet<Integer> contained = new HashSet<>();
        
        for(int i=0;i<numClausesRed;i++){
                r1 = rand.nextInt(first.size()+1);
                if(contained.contains(r1)){
                    i--;
                    continue;
                }
            if(r1 != first.size()){
                contained.add(r1);
                tmp.redNodes.get(0).set(i,first.get(r1));
            } 
        }
        
        contained.clear();
        
        for(int i=0;i<numClausesRed;i++){
                r1 = rand.nextInt(second.size()+1);
                if(contained.contains(r1)){
                    i--;
                    continue;
                }
            if(r1 != second.size()){
                contained.add(r1);
                tmp.redNodes.get(1).set(i,second.get(r1));
            } 
        }
        contained.clear();
        
        tmp.computeEntities(map);
        tmp.computeStats(data);
        
        RedescriptionScore rs = new RedescriptionScore();
        
         double initialscore = 0;
            double laterscore = 0;
            Redescription tmp1 = null;
             size = first.size()+second.size()+2;
             double scores[] = new double[2];
            double explogprobabilities[] = null;
              Random unif = new Random();
        //chose one leaf at random from each tree and add to redescription queries
        while(iteration < MaxIterationRed && !equilibrium){
            
           initialscore = rs.computeScore(tmp, this.minSupport, this.maxSupport, this.minimalJaccard, this.maxPval);

           //add algo
           //create a copy of a redescription
           tmp1 = copyRedescriptionNodes(tmp);
           //chose leaf at random
           r1 = rand.nextInt(size);
           
           //chose position at random, position <numClauses
            qn = rand.nextInt(numClausesRed);
           
           //create a function modify(Clause c, position i)
           
           System.out.println("First size: "+first.size());
           System.out.println("Secod size: "+second.size());
           System.out.println("Chosen clause: "+qn);
           System.out.println("Chosen index: "+r1);
           
            if(r1<=first.size()){
               if(r1 == first.size()){//select an index and remove the node
                   tmp1.redNodes.get(0).set(qn, null);
                   System.out.println("Nullified!");
               }
               else{
                   tmp1.redNodes.get(0).set(qn, first.get(r1));
                   System.out.println("Q: ");
                   System.out.println(tmp1.createElConjunctionString(first.get(r1)));
               }
           }
           else{
               if(r1 == (size-1)){//select the index and remove the node
                   tmp1.redNodes.get(1).set(qn, null);
                    System.out.println("Nullified!");
               }
               else{
                   tmp1.redNodes.get(1).set(qn, second.get(r1-first.size()-1));
                   System.out.println("Q: ");
                   System.out.println(tmp1.createElConjunctionString(second.get(r1-first.size()-1)));
               }
           }
           
            tmp1.computeEntities(map);
            tmp1.computeStats(data);
            
            if(tmp1.support >=this.maxSupport){
                //tmp1.normalizeRedescription();
                /*tmp1.queryStrings.add(tmp1.createQueryString(0));
                tmp1.queryStrings.add(tmp1.createQueryString(1));
                System.out.println("Sup: "+tmp1.support);
                System.out.println(tmp1.queryStrings.get(0));
                System.out.println(tmp1.queryStrings.get(1));
                System.out.println("Max supp: "+this.maxSupport);
                System.out.println();*/
               // iteration++;
                //continue;
            }
            
           laterscore = rs.computeScore(tmp1, this.minSupport, this.maxSupport, this.minimalJaccard, this.maxPval);
           //compute new score on a copy 
            //iterate, add remove change with a predefined score
            //change with Gumber trick
           // double initialpro = expProbability(initialscore,splitmarkovbudget);
            //double laterpro = expProbability(laterscore,splitmarkovbudget);
            
             scores[0] = initialscore; scores[1] = laterscore;
                        
                        explogprobabilities = expLogProbability(scores, splitmarkovbudget); 
                        
                        
                         double gs[] = new double[2];
                
                            for(int ig=0;ig<gs.length;ig++){
                                double rn = unif.nextDouble();
                                gs[ig] = -Math.log(-Math.log(rn));
                               // if(gs[ig]<0)
                                 //   gs[ig] = 0.0;
                            }
			
                            if(gs[0] < 0 && gs[1] < 0){
                                double tmpGT = Math.abs(gs[0]);
                                gs[0] = Math.abs(gs[1]);
                                gs[1] = Math.abs(tmpGT);
                            }
                            else if(gs[0]<0 && gs[1]>=0){
                                gs[0] = 0;
                            }
                            else if(gs[1]<0 && gs[0]>=0){
                                gs[1] = 0;
                            }
                            
        System.out.println("log scores: ");
        System.out.println(explogprobabilities[0]+" "+explogprobabilities[1]);
        
        double initialpro = explogprobabilities[0]+gs[0];
        double laterpro = explogprobabilities[1]+gs[1];
        
       // System.out.println("Init prob: "+initialpro);
        //System.out.println("Later prob: "+laterpro);
          
			double ratio = (double)(laterpro/initialpro); 
                        
                        System.out.println("Ratio: "+ratio);
                        
                        if(initialpro == 0)
                            ratio = 1.0;
				
			//double ratio = (double)(laterpro/initialpro);         
			if(ratio>=1){
                            System.out.println("Replacing ratio>1: ");
                                    System.out.println("Previous Jaccard: "+tmp.Jaccard);
                                    System.out.println("New Jaccard: "+tmp1.Jaccard);
                                    System.out.println("Previous p-val: "+tmp.pval);
                                    System.out.println("New p-val: "+tmp1.pval);
                                    System.out.println("New support: "+tmp1.support);
                           tmp = tmp1;

				double totalscore = rs.computeScore(tmp, this.minSupport ,this.maxSupport, this.minimalJaccard, this.maxPval); //change        
				equilibrium = isEquilibriumRed(totalscore, tmp);          
			}
			else{
				boolean replace = false;
				double randomdouble = Random.nextDouble();
				if(randomdouble<ratio){
					replace = true;
				}
                                
                                System.out.println("Replace: "+replace);
                                
				if(replace){
                                    System.out.println("Replacing: ");
                                    System.out.println("Previous Jaccard: "+tmp.Jaccard);
                                    System.out.println("New Jaccard: "+tmp1.Jaccard);
                                    System.out.println("Previous p-val: "+tmp.pval);
                                    System.out.println("New p-val: "+tmp1.pval);
                                    System.out.println("New support: "+tmp1.support);

                                    tmp = tmp1;

				double totalscore = rs.computeScore(tmp, this.minSupport, this.maxSupport, this.minimalJaccard,this.maxPval); //change        
				equilibrium = isEquilibriumRed(totalscore,tmp);          
				}
				else{
                                    
				}
			}

        	iteration++;      
               System.out.println("Initial score: "+initialscore);
               System.out.println("Final score: "+laterscore);
               System.out.println("Iteration red creation: "+iteration);
        }
        
       
        tmp.normalizeRedescription();
        
        if(tmp.redNodes.get(0).size() == 0){
            tmp.redNodes.get(0).add(first.get(rand.nextInt(first.size())));
        }
        
        if(tmp.redNodes.get(1).size() == 0){
            tmp.redNodes.get(1).add(second.get(rand.nextInt(second.size())));
        }

        tmp.computeEntities(map);
        tmp.computeStats(data);
        tmp.queryStrings.add(tmp.createQueryString(0));
        tmp.queryStrings.add(tmp.createQueryString(1));
        System.out.println("Final red Jaccard: "+tmp.Jaccard);
        System.out.println("Final red support: "+tmp.support);
        System.out.println("Final red p-value: "+tmp.pval);
        //return constructed redescription
        return tmp;
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
         
        //epsilon must be divided by 2
         BigDecimal budg = this.noisebudget.divide(new BigDecimal(2.0));
       
      
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
                    else if(selection == 1){
                        usedW1.add(i+1); usedW2.add(-(j+1));
                    }
                    else if(selection == 1){
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
