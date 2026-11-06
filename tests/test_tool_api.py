from types import SimpleNamespace
import unittest
from weight_programming import ForwardPolicy
from weight_programming.tool_api import tool_schemas,dispatch_tool


class Session:
    service=SimpleNamespace(backend=SimpleNamespace(num_layers=24,full_attention_layers=(3,7,11,15,19,23)),policy=ForwardPolicy())
    def execute(self,code):return ('python',code)
    def forward(self,request):return ('forward',request)


class ToolAPITests(unittest.TestCase):
    def test_schema_encodes_hybrid_sites_and_dispatch_rejects_extensions(self):
        import jsonschema
        session=Session();schemas=tool_schemas(session)
        for tool in schemas:jsonschema.Draft202012Validator.check_schema(tool['parameters'])
        forward=schemas[1]['parameters']
        jsonschema.validate({'type':'teacher_forced','model':'original','texts':['x'],'return':{'hidden_states':[0,24],'attn_layers':[3]}},forward)
        for bad in [
            {'type':'teacher_forced','model':'both','texts':['x']},
            {'type':'teacher_forced','model':'original','texts':['x'],'return':{'attn_layers':[0]}},
            {'type':'generate','model':'original'},
            {'type':'activation_patch','model':'original','texts':['x']}]:
            with self.assertRaises(jsonschema.ValidationError):jsonschema.validate(bad,forward)
        self.assertEqual(dispatch_tool(session,'execute_python',{'code':'1+1'}),('python','1+1'))
        with self.assertRaises(ValueError):dispatch_tool(session,'activation_patch',{})
        with self.assertRaises(ValueError):dispatch_tool(session,'execute_python',{'code':'x','budget':999})


if __name__=='__main__':unittest.main()
