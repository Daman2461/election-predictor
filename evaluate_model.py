import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd

# Test data with ground truth labels
# Format: (text, label) where label is 0 (left), 1 (center), or 2 (right)
TEST_DATA = [
    # Left-leaning articles (label: 0)
    ("""
    The urgent need for comprehensive healthcare reform has never been more apparent. With millions of Americans still uninsured and medical costs skyrocketing, 
    a single-payer healthcare system represents the most equitable solution. Studies show that countries with universal healthcare achieve better health outcomes 
    at lower costs. The current profit-driven healthcare system prioritizes corporate interests over patient care, leading to exorbitant prescription drug prices 
    and unnecessary medical procedures. We must prioritize healthcare as a fundamental human right, not a privilege reserved for those who can afford it. 
    The implementation of Medicare for All would eliminate private insurance overhead costs, reduce administrative waste, and ensure that every American 
    has access to quality healthcare regardless of their income or employment status. Furthermore, a single-payer system would give the government the 
    bargaining power to negotiate lower drug prices and reduce the influence of pharmaceutical companies on healthcare policy.
    """, 0),

    ("""
    Climate change represents an existential threat that requires immediate and radical action. The scientific consensus is clear: we must reduce carbon emissions 
    by at least 50% by 2030 to avoid catastrophic consequences. This necessitates a complete transition to renewable energy sources, massive investment in 
    public transportation, and strict regulations on corporate polluters. The Green New Deal provides a comprehensive framework for addressing both the climate 
    crisis and economic inequality. We must hold fossil fuel companies accountable for their role in climate change and ensure a just transition for workers 
    in affected industries. The time for incremental change has passed; we need bold, transformative action to protect our planet and future generations. 
    This includes implementing a carbon tax, ending fossil fuel subsidies, and investing heavily in green infrastructure and technology. We must also 
    address environmental justice issues, as climate change disproportionately affects marginalized communities.
    """, 0),

    ("""
    The growing wealth gap in America demands immediate attention and action. While corporate profits and CEO salaries reach record highs, millions of workers 
    struggle to make ends meet on minimum wage. The current economic system disproportionately benefits the wealthy while leaving working families behind. 
    We need progressive tax reform that ensures the ultra-rich pay their fair share, a significant increase in the minimum wage, and stronger labor protections. 
    The implementation of a wealth tax on billionaires and increased capital gains taxes would help fund essential social programs and reduce economic inequality. 
    Additionally, we must strengthen unions and collective bargaining rights to give workers more power in negotiating fair wages and benefits. The recent 
    trend of stock buybacks and corporate tax avoidance must be addressed through stricter regulations and enforcement. We should also implement policies 
    that promote worker ownership and cooperative business models to create more equitable economic structures.
    """, 0),

    ("""
    The criminal justice system in America is in desperate need of reform. Mass incarceration has created a modern-day system of racial and economic oppression, 
    with communities of color disproportionately affected. The war on drugs has failed, leading to overcrowded prisons and broken families. We need to end 
    cash bail, which criminalizes poverty, and implement alternatives to incarceration for non-violent offenses. Police reform must include mandatory body 
    cameras, independent oversight boards, and stricter accountability measures. The private prison industry should be abolished, as it creates perverse 
    incentives for incarceration. We must also address the school-to-prison pipeline by investing in education and social services instead of punitive measures. 
    The focus should be on rehabilitation and restorative justice rather than punishment and retribution.
    """, 0),

    ("""
    Education is a fundamental right that should be accessible to all Americans, regardless of their economic background. The current system of student debt 
    has created a generation of young people burdened with financial obligations that limit their life choices and economic mobility. We need to make public 
    colleges and universities tuition-free and cancel existing student debt. Additionally, we must address the racial and economic segregation in our K-12 
    schools by increasing funding for public schools in low-income areas and implementing policies that promote integration. The influence of standardized 
    testing should be reduced, and more emphasis should be placed on critical thinking and creative problem-solving. We should also invest in early childhood 
    education and provide universal pre-K to give all children a strong foundation for learning.
    """, 0),

    # Center-leaning articles (label: 1)
    ("""
    The debate over healthcare reform requires a balanced approach that considers both market forces and public needs. While the current system has significant 
    flaws, a complete overhaul to single-payer healthcare may not be the most practical solution. Instead, we should focus on improving the Affordable Care Act, 
    expanding Medicaid in states that haven't done so, and implementing cost-control measures. Public-private partnerships could help reduce prescription drug 
    prices while maintaining innovation in pharmaceutical research. We must also address the root causes of high healthcare costs, including administrative 
    inefficiencies and the lack of price transparency. A pragmatic approach that combines market-based solutions with targeted government intervention 
    could achieve better outcomes for all Americans. This might include a public option that competes with private insurance while preserving consumer choice.
    """, 1),

    ("""
    Addressing climate change requires a balanced approach that considers both environmental protection and economic growth. While reducing carbon emissions 
    is crucial, we must ensure that the transition to renewable energy doesn't disproportionately impact workers and communities dependent on traditional 
    energy sectors. A combination of market-based solutions, such as carbon pricing, and government incentives for clean energy development could accelerate 
    the transition while maintaining economic stability. International cooperation is essential, as climate change is a global challenge that requires 
    coordinated action. We should invest in research and development of new technologies while implementing practical, achievable emissions reduction targets. 
    The focus should be on innovation and adaptation rather than drastic economic disruption. This includes supporting the development of carbon capture 
    technology and nuclear energy as part of a diverse energy portfolio.
    """, 1),

    ("""
    The minimum wage debate highlights the complex balance between supporting workers and maintaining business competitiveness. While workers deserve fair 
    compensation, sudden, dramatic increases in the minimum wage could have unintended consequences for small businesses and employment levels. A more 
    nuanced approach might involve regional minimum wages that account for local cost of living, combined with tax credits for low-income workers and 
    targeted support for small businesses. We should also focus on improving worker skills and education to help them qualify for higher-paying jobs. 
    The goal should be to ensure workers can earn a living wage while maintaining a healthy business environment that supports job creation. This might 
    include implementing a gradual, phased approach to minimum wage increases and providing additional support for small businesses during the transition.
    """, 1),

    ("""
    The debate over immigration policy highlights the need for comprehensive reform that balances security concerns with economic realities. While border 
    security is essential, we must also recognize the contributions of immigrants to our economy and society. A balanced approach would include enhanced 
    border security measures, a streamlined path to legal immigration, and reforms to the visa system to better meet labor market needs. We should also 
    address the root causes of illegal immigration through international cooperation and economic development programs. The goal should be a system that 
    is both secure and welcoming to those who wish to contribute to our nation's success. This includes providing a path to citizenship for Dreamers and 
    implementing a more efficient guest worker program that meets the needs of both employers and workers.
    """, 1),

    ("""
    The role of government in regulating big tech companies has become a pressing issue in our digital age. While these companies have driven innovation 
    and economic growth, concerns about privacy, market dominance, and content moderation have raised legitimate questions about the need for oversight. 
    A balanced approach would involve targeted regulations that protect consumer privacy and prevent anti-competitive practices while avoiding excessive 
    government intervention that could stifle innovation. We should focus on creating clear rules of the road that allow the tech industry to continue 
    growing while addressing legitimate concerns about market power and user protection. This might include updating antitrust laws for the digital age 
    and implementing stronger data privacy protections while preserving the benefits of technological innovation.
    """, 1),

    ("""
    The future of work in America requires careful consideration of both technological advancement and worker protection. While automation and artificial 
    intelligence will transform many industries, we must ensure that workers are not left behind. A balanced approach would include investment in education 
    and retraining programs, along with policies that support job creation in emerging sectors. We should also consider how to adapt our social safety net 
    to address the changing nature of work, including the growth of the gig economy. The goal should be to harness technological progress while ensuring 
    that all workers can participate in and benefit from economic growth. This might include exploring universal basic income pilots and implementing 
    portable benefits for workers in non-traditional employment arrangements.
    """, 1),

    # Right-leaning articles (label: 2)
    ("""
    The push for government-run healthcare represents a dangerous step toward socialism that would undermine the quality of American healthcare. The current 
    system, while imperfect, has produced the world's most advanced medical technology and treatments. Instead of a complete government takeover, we should 
    focus on market-based reforms that increase competition and reduce costs. This includes allowing insurance sales across state lines, expanding health 
    savings accounts, and implementing tort reform to reduce defensive medicine. The private sector's role in healthcare innovation must be preserved, as 
    government control would stifle medical advancement and lead to rationing of care. We need solutions that protect patient choice and maintain the 
    high standards of American healthcare. This includes promoting price transparency and encouraging direct primary care models that bypass insurance 
    bureaucracy.
    """, 2),

    ("""
    The climate change agenda threatens American energy independence and economic growth. While environmental protection is important, the proposed Green New 
    Deal would devastate our economy and cost millions of jobs. Instead of radical, government-mandated changes, we should focus on innovation and market-based 
    solutions. American energy companies have already made significant progress in reducing emissions through technological advancement. We should continue 
    to develop all energy sources, including clean coal, nuclear power, and natural gas, while removing unnecessary regulations that hinder energy production. 
    The United States should maintain its energy independence and resist international agreements that would put us at an economic disadvantage. This includes 
    supporting the development of new technologies that make fossil fuels cleaner and more efficient while maintaining our competitive edge in energy production.
    """, 2),

    ("""
    The minimum wage debate ignores basic economic principles and the realities of running a business. Government-mandated wage increases would force many 
    small businesses to close or reduce their workforce, ultimately hurting the very workers they aim to help. Instead of artificial wage controls, we should 
    focus on creating an economic environment that allows businesses to grow and create more high-paying jobs. This includes reducing business regulations, 
    lowering corporate taxes, and promoting free trade. The path to higher wages lies in economic growth and increased productivity, not government mandates. 
    We should also emphasize the importance of education and skills development to help workers qualify for better-paying positions. This includes expanding 
    vocational training and apprenticeship programs that provide practical skills for in-demand jobs.
    """, 2),

    ("""
    The push for increased government regulation of the economy threatens American prosperity and individual freedom. While some oversight is necessary, 
    excessive regulation stifles innovation and economic growth. We should focus on reducing regulatory burdens, simplifying the tax code, and promoting 
    free market competition. The private sector, not government, is the engine of economic growth and job creation. We need policies that encourage 
    entrepreneurship and investment, including lower corporate tax rates and reduced capital gains taxes. The role of government should be to create a 
    level playing field and enforce contracts, not to pick winners and losers in the marketplace. This includes rolling back unnecessary regulations 
    and implementing regulatory reform that requires cost-benefit analysis for new rules.
    """, 2),

    ("""
    The Second Amendment is a fundamental right that must be protected from government overreach. While reasonable measures to prevent gun violence are 
    important, the focus should be on enforcing existing laws and addressing the root causes of violence, not restricting the rights of law-abiding 
    citizens. The right to self-defense is inherent and should not be subject to government permission. We should focus on improving mental health care, 
    strengthening school security, and prosecuting violent criminals rather than implementing new gun control measures. The solution to gun violence lies 
    in addressing the breakdown of family and community values, not in restricting constitutional rights. This includes supporting programs that promote 
    responsible gun ownership and proper firearm training.
    """, 2),

    # Additional mixed-content articles (label: 1)
    ("""
    The debate over social media regulation highlights the complex balance between free speech and public safety. While these platforms have revolutionized 
    communication and enabled unprecedented access to information, concerns about misinformation, privacy, and mental health have raised important questions 
    about their role in society. A balanced approach would involve targeted regulations that address specific harms while preserving the benefits of these 
    platforms. This includes improving transparency in content moderation, protecting user privacy, and addressing the impact of social media on mental health, 
    particularly among young people. We should focus on creating a regulatory framework that promotes responsible innovation while protecting users from 
    clear harms. This might include updating Section 230 to better address modern challenges while preserving the open nature of the internet.
    """, 1),

    ("""
    The future of American infrastructure requires a balanced approach that considers both immediate needs and long-term sustainability. While our roads, 
    bridges, and utilities need significant investment, we must ensure that these projects are cost-effective and environmentally responsible. A comprehensive 
    infrastructure plan should include traditional projects like road and bridge repair, as well as modern priorities like broadband expansion and clean 
    energy infrastructure. The focus should be on public-private partnerships that leverage private investment while maintaining public oversight. We should 
    also prioritize projects that create good-paying jobs and promote economic growth in underserved communities. This includes implementing streamlined 
    permitting processes and ensuring that infrastructure investments are distributed fairly across urban and rural areas.
    """, 1),

    ("""
    The debate over artificial intelligence and automation requires careful consideration of both technological progress and human impact. While these 
    technologies offer tremendous potential for economic growth and improved quality of life, they also raise important questions about job displacement 
    and ethical considerations. A balanced approach would involve supporting innovation while implementing policies that help workers adapt to changing 
    labor markets. This includes investing in education and training programs that prepare workers for the jobs of the future, as well as developing 
    ethical frameworks for AI development and deployment. We should focus on harnessing the benefits of automation while ensuring that the economic 
    gains are widely shared. This might include exploring new models of worker ownership and participation in the digital economy.
    """, 1)
]

def load_model():
    model_path =  "/Users/daman/Downloads/Article-Bias-Prediction-main/output/model2"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    # Use MPS (Metal Performance Shaders) for M-series Macs
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Metal Performance Shaders) for GPU acceleration")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA for GPU acceleration")
    else:
        device = torch.device("cpu")
        print("Using CPU for inference")
    
    model.to(device)
    model.eval()
    return model, tokenizer, device

def predict_bias(text, model, tokenizer, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return predicted_class

def evaluate_model():
    # Load model
    model, tokenizer, device = load_model()
    
    # Prepare data
    texts = [item[0] for item in TEST_DATA]
    true_labels = [item[1] for item in TEST_DATA]
    
    # Get predictions
    predictions = [predict_bias(text, model, tokenizer, device) for text in texts]
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='weighted')
    accuracy = accuracy_score(true_labels, predictions)
    
    # Print results
    print("\nModel Evaluation Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Print per-class metrics
    print("\nPer-class Metrics:")
    precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
        true_labels, predictions, labels=[0, 1, 2]
    )
    
    metrics_df = pd.DataFrame({
        'Class': ['Left', 'Center', 'Right'],
        'Precision': precision_per_class,
        'Recall': recall_per_class,
        'F1-Score': f1_per_class,
        'Support': support
    })
    
    print("\nDetailed Metrics:")
    print(metrics_df.to_string(index=False))
    
    # Print confusion matrix
    print("\nPredictions vs True Labels:")
    for text, true_label, pred_label in zip(texts, true_labels, predictions):
        label_map = {0: "Left", 1: "Center", 2: "Right"}
        print(f"Text: {text[:150]}...")
        print(f"True: {label_map[true_label]}, Predicted: {label_map[pred_label]}\n")

if __name__ == "__main__":
    evaluate_model() 