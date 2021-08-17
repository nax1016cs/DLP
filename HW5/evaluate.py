from train import *
def evaluate(generator, dataset, latent_size, name):
    evaluation_model = EvaluationModel()
    test_conditions = Condition(dataset).to(device)
    fz = sample(len(test_conditions), latent_size).to(device)
    generator.eval()
    with torch.no_grad():
        gen_imgs = generator(fz, test_conditions)
    score = evaluation_model.eval(gen_imgs, test_conditions)
    save_image(gen_imgs, os.path.join("results", name), nrow = 8, normalize = True)
    return score
