@Grapes([
          @Grab(group='org.semanticweb.elk', module='elk-owlapi', version='0.4.3'),
          @Grab(group='net.sourceforge.owlapi', module='owlapi-api', version='4.5.17'),
          @Grab(group='net.sourceforge.owlapi', module='owlapi-apibinding', version='4.5.17'),
          @Grab(group='net.sourceforge.owlapi', module='owlapi-impl', version='4.5.17'),
          @Grab(group='net.sourceforge.owlapi', module='owlapi-parsers', version='4.5.17'),
          @GrabConfig(systemClassLoader=true)
        ])

import org.semanticweb.owlapi.model.parameters.*
import org.semanticweb.elk.owlapi.ElkReasonerFactory;
import org.semanticweb.elk.owlapi.ElkReasonerConfiguration
import org.semanticweb.elk.reasoner.config.*
import org.semanticweb.owlapi.apibinding.OWLManager;
import org.semanticweb.owlapi.reasoner.*
import org.semanticweb.owlapi.reasoner.structural.StructuralReasoner
import org.semanticweb.owlapi.vocab.OWLRDFVocabulary;
import org.semanticweb.owlapi.model.*;
import org.semanticweb.owlapi.io.*;
import org.semanticweb.owlapi.owllink.*;
import org.semanticweb.owlapi.util.*;
import org.semanticweb.owlapi.search.*;
import org.semanticweb.owlapi.manchestersyntax.renderer.*;
import org.semanticweb.owlapi.reasoner.structural.*


OWLOntologyManager manager = OWLManager.createOWLOntologyManager()
OWLOntology ont = manager.loadOntologyFromOntologyDocument(new File("onto_list.owl"))
OWLDataFactory fac = manager.getOWLDataFactory()

OWLOntology outont = manager.createOntology(new IRI("http://phenomebrowser.net/up.owl"))

ont.getAxioms(org.semanticweb.owlapi.model.parameters.Imports.INCLUDED).each { ax ->
    manager.addAxiom(outont, ax)
}

manager.saveOntology(outont, IRI.create("file:/Users/josephtang/Desktop/train/pheno.owl"))
