package generaldpalgos;

import java.io.Serializable;
import java.math.BigDecimal;
import java.math.MathContext;
import java.math.RoundingMode;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;
import org.apache.commons.math3.util.Pair;



import weka.classifiers.AbstractClassifier;
import weka.core.*;
//exponential mechanism DP tree
/**
 *
 * This class is developed by the authors of the manuscript: "Embedding differential privacy in decision tree
algorithm with different depths" by authors Xuanyu BAI , Jianguo YAO*, Mingxuan YUAN, Ke DENG, Xike Xie and Haibing Guan.
*It contains measures for differentially private algorithms DecTreeWholeDP and DecTreeExpMech.
 */
public class DecTreeExpMech extends AbstractClassifier{

	private static final long serialVersionUID = 1L;    
        private double totalNumEntities;
      
    public DecTreeExpMech(){
        totalNumEntities = 0.0; 
    }
    
        
    public DecTreeExpMech(int numEntities){
        totalNumEntities = numEntities; 
    }

    public static final MathContext MATH_CONTEXT = new MathContext(20, RoundingMode.DOWN);   
    
	private int MaxDepth;
	public void setMaxDepth(int maxDepth) {
		MaxDepth = maxDepth;
	}
	
	private BigDecimal Epsilon = new BigDecimal(1.0);       
    public void setEpsilon(String epsilon) {                       
	    if (epsilon!=null && epsilon.length()!=0)
	        Epsilon = new BigDecimal(epsilon,MATH_CONTEXT);
    }
    private BigDecimal onelevelbudget = new BigDecimal(1.0,MATH_CONTEXT);
    private BigDecimal splitbudget = new BigDecimal(1.0,MATH_CONTEXT);
    private BigDecimal noisebudget = new BigDecimal(1.0,MATH_CONTEXT);

    private Random Random = new Random();  
    public void setSeed(int seed){                         
		Random = new Random(seed);
	}

    private double GiniSen = 2;       
    
	private Node Root;
        
        
        public Node getRoot(){
            return Root;
        }
	
	//private Set<Attribute> Attributes = new HashSet<Attribute>();
        private Set<Pair<Attribute, Double>> Attributes = new HashSet<Pair<Attribute,Double>>();
        int numLeafs = 0;
	GiniIndexDirectAdd gini = new GiniIndexDirectAdd();  
	
	
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
         
         public int countLeafs(Node node, int count){
            
            if(node.isLeaf){
                count = count+1;
                return count;
            }
            
            for(int i=0;i<node.children.length;i++)
                count = countLeafs(node.children[i],count);
            
            return count;
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
         
         public Instances addTargets(Instances data){
            Instances tmp = new Instances(data);
            int classIndex = data.classIndex();
            
            tmp.setClassIndex(-1);
            tmp.deleteAttributeAt(classIndex);
            int cl = countLeafs(Root, 0);
        
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
             tmp = setTargets(Root, classIndex, tmp, instInNode);
             
            return tmp;
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
                 if(a == data.classAttribute())
                     continue;
                
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

	public void buildClassifier(Instances data) throws Exception {
        HashSet<Pair<Attribute,Double>> allAttributesGlobal = this.createAttrList(data, 0.8);
	HashSet<Pair<Attribute,Double>> allAttributes = new HashSet<>();
        Enumeration<Attribute> attrEnum = data.enumerateAttributes();  
        for(Pair<Attribute,Double> p:allAttributesGlobal)
                allAttributes.add(p);
        
        Attributes = new HashSet<Pair<Attribute,Double>>(allAttributes);

        Root = new Node(data, null, 0);                      
       
        onelevelbudget = Epsilon.divide(BigDecimal.valueOf(MaxDepth),MATH_CONTEXT);    
        System.out.println("Budget exp: "+onelevelbudget);
        splitbudget = onelevelbudget;
        noisebudget = onelevelbudget;                        
        
         Set<Pair<Attribute,Double>> Attrs = new HashSet<Pair<Attribute,Double>>(Attributes);
        partitionOne(Root, Attrs);   
        //partitionOneLP(Root, Attrs);

        //addNoise(Root, noisebudget);                               

	}
	
    private void partitionOne(Node node, Set<Pair<Attribute,Double>> Attrs){

    	Attrs = new HashSet<Pair<Attribute,Double>>(Attributes);
        Object atList[] = null;
        atList = Attrs.toArray();
        
        
    	/*if(node.data.numInstances()==0){
			makeLeafNode(node);
			return;
		}*/ //this can not be used - it breakes privacy
		
		/*for(int i=0; i< node.count.length; i++){        
			if(node.count[i] == node.data.numInstances()){
				makeLeafNode(node);
				return;
			}
		}*/ //this can not be used either!

		deleteParAttr(node,Attrs);         
		if(Attrs.size()==0){//OK, we used up all the attributes
			makeLeafNode(node);
			return;
		}

		if(node.depth >= MaxDepth){//OK max depth reached
			makeLeafNode(node);
			return;
		}
		
		makeInnerNode(node);

		double attrscores[] = new double[Attrs.size()];             
		double expprobabilities[] = new double[Attrs.size()];      
	
		for(int i=0; i<Attrs.size(); i++){//if numeric do this for every splitting point of this attribute
			
			Node tempNode = new Node(node);
			
			Pair<Attribute,Double> attr1 = (Pair<Attribute,Double>)atList[i];//Attrs.toArray()[i];      
                        Attribute splitAttr = attr1.getKey();//(Attribute)Attrs.toArray()[Random.nextInt(Attrs.size())];  
		       double splitValue = attr1.getValue();
			tempNode.children = null;
			
			Instances[] tempParts = partitionByAttr(tempNode.data,splitAttr,splitValue);
	     	       Node[] tempChildren = null; 
                       
                       if(splitAttr.isNumeric())
                            tempChildren = new Node[2];
                        else
                            tempChildren = new Node[splitAttr.numValues()];  

			for(int k=0; k < tempParts.length; k++){
				tempChildren[k] = new Node(tempParts[k], tempNode, k); 
				tempChildren[k].isLeaf = true;
			}
			tempNode.splitAttr = splitAttr;
                        tempNode.splitValue = splitValue;
			tempNode.children = tempChildren;
		
	    	//attrscores[i] = gini.score(tempNode,totalNumEntities);
                attrscores[i] = gini.score(tempNode); //save the best score for the attribute (among all splitting points) and the chosen splitting point
			expprobabilities[i] = expProbability(attrscores[i], splitbudget);         
		}
                
               /* System.out.println("Scores and probabilities: ");
                for(int zj = 0; zj<attrscores.length;zj++)
                    System.out.print(attrscores[zj]+" ");
                System.out.println();
                
                for(int zj = 0; zj<expprobabilities.length;zj++)
                    System.out.print(expprobabilities[zj]+" ");
                
                System.out.println();*/
		
		if(Utils.sum(expprobabilities) != 0){                    
		    Utils.normalize(expprobabilities);         
		}
                
               /* System.out.println("Normalized probabilities: ");
                
                 for(int zj = 0; zj<expprobabilities.length;zj++)
                    System.out.print(expprobabilities[zj]+" ");
                
                System.out.println();*/

		for(int j=expprobabilities.length-1; j>=0; j--){
			double sum = 0;
			for(int k=0; k<=j; k++){
				sum += expprobabilities[k];
			}
			expprobabilities[j] = sum;
		}
                
                /* System.out.println("Cumulative probabilities: ");
                
                 for(int zj = 0; zj<expprobabilities.length;zj++)
                    System.out.print(expprobabilities[zj]+" ");
                
                System.out.println();
	*/
		double randouble = Random.nextDouble();  
               // System.out.println("Random number: "+randouble);
                
        int flag = 0;
        if(randouble < expprobabilities[0]){
        	flag = 0;
        }
        else{
        	 for(int t=0; t<expprobabilities.length-1; t++){
             	if((randouble >= expprobabilities[t])&&(randouble < expprobabilities[t+1])){
             		flag = t+1;
             	}
             }
        }
        
        /*System.out.println("Flag: "+flag);
        System.out.println();*/
        
            Pair<Attribute,Double> goalAttr = (Pair<Attribute,Double>)atList[flag];//(Attribute)Attrs.toArray()[flag];
            Attribute splitAttr = goalAttr.getKey();//(Attribute)Attrs.toArray()[Random.nextInt(Attrs.size())];  
            double splitValue = goalAttr.getValue();
	    Instances[] parts = partitionByAttr(node.data, splitAttr, splitValue);  
            
	    Node[] children = null;      
            
            if(splitAttr.isNumeric())
                            children = new Node[2];
                        else
                            children = new Node[splitAttr.numValues()];  

	    node.children = children;              
	    node.splitAttr = splitAttr;  
            node.splitValue = splitValue;
			
		for(int k=0; k < parts.length; k++){
			children[k] = new Node(parts[k], node, k);   
			
            partitionOne(children[k], Attrs);
		}
	}
	
	private void makeLeafNode(Node node){
		
		node.splitAttr = null;                                          
		node.children  = null;                                    
		node.isLeaf = true;	
	}
	
	private void makeInnerNode(Node node){

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
    	double beta = bigBeta.doubleValue();                
    	
        double uniform = Random.nextDouble()-0.5;            
        return miu-beta*((uniform>0) ? -Math.log(1.-2*uniform) : Math.log(1.+2*uniform)); 
        
    }

    private double[] expLogProbability(double [] score, BigDecimal epsilon)              
  	{
                double res[] = new double[score.length];
                
                for(int i=0;i<res.length;i++)
                    res[i] = epsilon.doubleValue() * score[i] / (2 * GiniSen);
  		//return Math.exp(epsilon.doubleValue() * score / (2 * GiniSen));
                return res;
  	}  
    
  	private double expProbability(double score, BigDecimal epsilon)              
  	{
  		return Math.exp(epsilon.doubleValue() * score / (2 * GiniSen));
  	}                  
  	
    public double classifyInstance(Instance instance) throws NoSupportForMissingValuesException { 	
    	
    	assert( instance.hasMissingValue() == false);
    	
    	Node node = Root;                                       
    	while(node.isLeaf == false){                     
    		Attribute attr = node.splitAttr;
    		node = node.children[ (int)instance.value(attr) ];
    	}
    	
    	return Utils.maxIndex(node.dist);         
    }

	public double[] distributionForInstance(Instance instance) throws NoSupportForMissingValuesException {	
		
		assert( instance.hasMissingValue() == false);
		
		Node node = Root;                                 
    	while(node.isLeaf == false){                      
    		Attribute attr = node.splitAttr;
    		node = node.children[ (int)instance.value(attr) ];
    	}
    	
    	return node.dist;                          
	}                                       
	
    public String toString() {                    

        return toString(Root);
       
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
